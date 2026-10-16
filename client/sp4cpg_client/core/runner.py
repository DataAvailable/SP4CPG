
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from .config import ClientConfig


@dataclass
class StepResult:
    name: str
    command: List[str]
    cwd: str
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class WorkflowResult:
    ok: bool
    mode: str
    output_dir: str
    steps: List[StepResult] = field(default_factory=list)
    metrics: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    result_table_path: str = ""
    message: str = ""


class ProcessStreamer(QObject):
    line = Signal(str)
    step_started = Signal(str, str)
    step_finished = Signal(str, bool, int)
    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ClientConfig, mode: str):
        super().__init__()
        self.config = config
        self.mode = mode
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            result = WorkflowRunner(self.config, self.mode, self).run()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class WorkflowThread(QThread):
    def __init__(self, config: ClientConfig, mode: str):
        super().__init__()
        self.streamer = ProcessStreamer(config, mode)
        self.streamer.moveToThread(self)
        self.started.connect(self.streamer.run)

    def stop(self) -> None:
        self.streamer.stop()


class WorkflowRunner:
    def __init__(self, config: ClientConfig, mode: str, emitter: Optional[ProcessStreamer] = None):
        self.config = config
        self.mode = mode
        self.emitter = emitter
        self.out = config.resolved_output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.result = WorkflowResult(ok=False, mode=mode, output_dir=str(self.out))

    def run(self) -> WorkflowResult:
        self._validate_repo()
        env = self._build_env()
        steps = self._build_steps()
        if not steps:
            raise ValueError("没有可执行步骤，请检查任务配置。")
        total = len(steps)
        for idx, (name, command, cwd) in enumerate(steps, start=1):
            if self.emitter and self.emitter._stop:
                self.result.message = "任务已由用户取消。"
                return self.result
            self._emit_step_start(name, command)
            step = self._run_command(name, command, cwd, env)
            self.result.steps.append(step)
            ok = step.returncode == 0
            self._emit_step_finish(name, ok, step.returncode)
            self._emit_progress(int(idx * 100 / total))
            if not ok:
                self.result.message = f"步骤失败：{name}，返回码 {step.returncode}。"
                self._collect_artifacts()
                return self.result
        self._collect_artifacts()
        self._parse_metrics()
        self.result.ok = True
        self.result.message = "HGCN + MEDE 检测流程执行完成。"
        return self.result

    def _validate_repo(self) -> None:
        repo = self.config.repo
        required = [repo / "data"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("项目路径无效，缺少：" + ", ".join(missing))
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"数据集不存在：{self.config.data_path}")

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        repo = str(self.config.repo)
        client_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repo + os.pathsep + client_root + os.pathsep + env.get("PYTHONPATH", "")
        if self.config.joern_home:
            env["JOERN_HOME"] = self.config.joern_home
            env["PATH"] = self.config.joern_home + os.pathsep + env.get("PATH", "")
        if self.config.MEDE_home:
            env["MEDE_HOME"] = self.config.MEDE_home
            env["PATH"] = self.config.MEDE_home + os.pathsep + env.get("PATH", "")
        if self.config.skills_home:
            env["SKILLS_HOME"] = self.config.skills_home
        if self.config.llm_config_path:
            env["LLM_CONFIG"] = self.config.llm_config_path
        return env

    def _build_steps(self) -> List[Tuple[str, List[str], str]]:
        python = self.config.python_exe.strip() if self.config.python_exe else sys.executable
        if python.lower() == "python":
            python = sys.executable
        repo = self.config.repo
        pp = self.config.preprocess_dir
        tools = Path(__file__).resolve().parents[1] / "tools"
        out = self.out
        data = self.config.data_path
        source_data = out / "artifacts" / "cpg" / "samples.jsonl" if self.config.run_cpg else data
        hgcn_data = self.config.hgcn_dataset_path
        steps: List[Tuple[str, List[str], str]] = []

        # Revision data uses explicit incidence and a versioned .pt container.
        if self.config.run_cpg:
            steps.append(("可选：CPG 图生成", [python, "-u", "cpg_generate.py", "--input", str(data),
                           "--output", str(out / "artifacts" / "cpg")], str(pp)))
        if self.config.run_hcpg:
            cpg_graphs = out / "artifacts" / "cpg" / "graphs" if self.config.run_cpg else pp / "dots-cpg"
            steps.append(("可选：HCPG 图构建", [python, "-u", "hcpg_generate.py", "--input", str(cpg_graphs),
                           "--output", str(out / "artifacts" / "hcpg"), "--format", "dot"], str(pp)))
        if self.config.run_embedding:
            hcpg_graphs = out / "artifacts" / "hcpg" if self.config.run_hcpg else pp / "dots-hcpg"
            embed = [python, "-u", "dot_embedding.py", "--input", str(hcpg_graphs), "--output", str(hgcn_data)]
            if self.config.run_cpg:
                embed += ["--labels", str(out / "artifacts" / "cpg" / "labels.json")]
            steps.append(("可选：HCPG 嵌入生成", embed, str(pp)))

        model_name = self.config.hgcn_model_name.strip() or "TAF-HGCN"
        if model_name not in {"TAF-HGCN", "HGNN", "GCN", "GAT", "GIN", "GraphSAGE", "GGNN", "HGCN"}:
            raise ValueError(f"不支持的模型类型：{model_name}")
        checkpoint = self.config.hgcn_checkpoint.strip()
        if self.mode == "train":
            if not self.config.hgcn_split_manifest:
                raise ValueError("训练需要显式样本划分清单；请先运行 scripts/make_split.py 并配置 split manifest")
            steps.append((
                "HGCN 重新训练与评估",
                [
                    python, "-u", str(repo / "main.py"),
                    "--model", model_name,
                    "--data", str(hgcn_data),
                    "--split", self.config.hgcn_split_manifest,
                    "--output", str(out / "training"),
                    "--batch", str(self.config.hgcn_batch),
                    "--lr", str(self.config.hgcn_lr),
                    "--dropout", "0.4",
                    "--epoch", str(self.config.hgcn_epochs),
                    "--patience", str(self.config.hgcn_patience),
                ],
                str(repo),
            ))
        else:
            cmd = [
                python, "-u", str(tools / "predict.py"),
                "--repo", str(repo),
                "--model", model_name,
                "--checkpoint", checkpoint,
                "--data", str(hgcn_data),
                "--output", str(out / "model_predictions.json"),
                "--csv", str(out / "model_predictions.csv"),
                "--metrics", str(out / "hgcn_metrics.json"),
                "--batch", str(self.config.hgcn_batch),
                "--dropout", "0.4",
            ]
            if self.config.allow_fallback:
                cmd += ["--allow-fallback", "--fallback-data", str(source_data)]
            if self.config.run_cpg or source_data.suffix.lower() in {".json", ".jsonl"}:
                cmd += ["--source-data", str(source_data)]
            steps.append(("HGCN 漏洞检测", cmd, str(repo)))

        # MEDE/static analysis stage. If user has a ready result, use it later; if not, run built-in detector.
        if self.mode == "full" or self.config.enable_MEDE:
            if self.config.MEDE_command.strip():
                steps.append(("MEDE 静态指针分析", self._split_command(self.config.MEDE_command), str(repo)))
            elif self.config.static_result_path.strip() and Path(self.config.static_result_path).exists():
                self._emit_line(f"[MEDE] 使用已有静态分析结果：{self.config.static_result_path}")
            else:
                allow = ["--allow-fallback"] if self.config.allow_fallback else []
                steps.append((
                    "MEDE 静态指针分析",
                    [
                        python, "-u", str(tools / "svf_detector.py"),
                        "--data", str(source_data),
                        "--bitcode", self.config.bitcode_path.strip(),
                        "--MEDE-home", self.config.MEDE_home.strip(),
                        "--output", str(out / "static_result.json"),
                        "--csv", str(out / "static_result.csv"),
                    ] + allow,
                    str(repo),
                ))

        if self.mode == "full":
            static_result = Path(self.config.static_result_path) if self.config.static_result_path else out / "static_result.json"
            steps.append((
                "交叉验证与综合判定",
                [
                    python, "-u", str(tools / "cross_validate.py"),
                    "--model-result", str(out / "model_predictions.json"),
                    "--static-result", str(static_result),
                    "--output", str(out / "final_report.json"),
                    "--csv", str(out / "final_report.csv"),
                ],
                str(repo),
            ))
        return steps

    def _split_command(self, cmd: str) -> List[str]:
        return shlex.split(cmd, posix=(os.name != "nt"))

    def _run_command(self, name: str, command: List[str], cwd: str, env: Dict[str, str]) -> StepResult:
        self._emit_line(f"\n===== {name} =====")
        self._emit_line(f"$ {self._format_command(command)}")
        stdout_lines: List[str] = []
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip("\n")
            stdout_lines.append(text)
            self._emit_line(text)
            if self.emitter and self.emitter._stop:
                proc.terminate()
                break
        returncode = proc.wait()
        return StepResult(
            name=name,
            command=command,
            cwd=cwd,
            returncode=returncode,
            stdout_tail="\n".join(stdout_lines[-160:]),
        )

    def _collect_artifacts(self) -> None:
        out = self.out
        repo = self.config.repo
        artifacts = {
            "数据集": self.config.data_path,
            "HGCN 模型目录": self.config.logs_dir,
            "HGCN 指标": out / "hgcn_metrics.json",
            "模型预测 JSON": out / "model_predictions.json",
            "模型预测 CSV": out / "model_predictions.csv",
            "MEDE 静态结果 JSON": out / "static_result.json",
            "MEDE 静态结果 CSV": out / "static_result.csv",
            "综合报告 JSON": out / "final_report.json",
            "综合报告 CSV": out / "final_report.csv",
            "CPG DOT 目录": repo / "preprocess" / "dots-cpg",
            "HCPG DOT 目录": repo / "preprocess" / "dots-hcpg",
        }
        self.result.artifacts = {k: str(v) for k, v in artifacts.items() if Path(v).exists()}
        if (out / "final_report.csv").exists():
            self.result.result_table_path = str(out / "final_report.csv")
        elif (out / "model_predictions.csv").exists():
            self.result.result_table_path = str(out / "model_predictions.csv")
        elif (out / "static_result.csv").exists():
            self.result.result_table_path = str(out / "static_result.csv")

    def _parse_metrics(self) -> None:
        metrics_path = self.out / "hgcn_metrics.json"
        metrics: Dict[str, str] = {}
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                for k in ["accuracy", "precision", "recall", "f1", "auc", "mode"]:
                    if k in data:
                        metrics[k] = str(data[k])
            except Exception:
                pass
        summary_path = self.out / "training" / "summary.json"
        if self.mode == "train" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for key, value in summary["evaluation"]["complete_test"].items():
                if key in {"accuracy", "precision", "recall", "f1", "samples"}:
                    metrics[key] = str(value)
        self.result.metrics = metrics

    def _format_command(self, command: List[str]) -> str:
        return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in command if str(x) != "")

    def _emit_line(self, text: str) -> None:
        if self.emitter:
            self.emitter.line.emit(text)

    def _emit_step_start(self, name: str, command: List[str]) -> None:
        if self.emitter:
            self.emitter.step_started.emit(name, self._format_command(command))

    def _emit_step_finish(self, name: str, ok: bool, code: int) -> None:
        if self.emitter:
            self.emitter.step_finished.emit(name, ok, code)

    def _emit_progress(self, value: int) -> None:
        if self.emitter:
            self.emitter.progress.emit(value)
