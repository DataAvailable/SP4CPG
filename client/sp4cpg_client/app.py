
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGraphicsDropShadowEffect, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget
)

from .core.config import APP_NAME, ClientConfig, load_config, save_config
from .core.runner import WorkflowResult, WorkflowThread
from .core.report import export_html_report


QSS = """
* { font-family: "Microsoft YaHei UI", "Segoe UI", Arial; font-size: 13px; }
QMainWindow, QWidget { background: #f6f8fc; color: #142033; }
QFrame#Sidebar { background:#ffffff; border-right:1px solid #dbe4f0; }
QFrame#Topbar { background:rgba(255,255,255,.98); border-bottom:1px solid #dbe4f0; }
QFrame#Hero { background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #eaf2ff, stop:.50 #ffffff, stop:1 #f4edff); border:1px solid #c7d2fe; border-radius:24px; }
QFrame#Card, QGroupBox { background:#ffffff; border:1px solid #dbe4f0; border-radius:18px; }
QGroupBox { margin-top:16px; padding:18px 16px 16px 16px; color:#0f172a; font-weight:850; }
QGroupBox::title { subcontrol-origin:margin; left:16px; padding:0 8px; color:#2563eb; background:#ffffff; }
QLabel#Muted { color:#64748b; }
QLabel#Tiny { color:#64748b; font-size:12px; }
QLabel#Title { color:#0f172a; font-size:24px; font-weight:900; }
QLabel#HeroTitle { color:#0f172a; font-size:28px; font-weight:950; }
QLabel#HeroText { color:#475569; font-size:14px; line-height:155%; }
QPushButton { background:#2563eb; color:#ffffff; border:0; border-radius:11px; padding:9px 16px; font-weight:800; }
QPushButton:hover { background:#1d4ed8; }
QPushButton:disabled { background:#cbd5e1; color:#64748b; }
QPushButton#Secondary { background:#ffffff; color:#334155; border:1px solid #d7e0ec; }
QPushButton#Secondary:hover { background:#eef4ff; color:#1d4ed8; border:1px solid #bfdbfe; }
QPushButton#Danger { background:#ef4444; color:white; }
QPushButton#Danger:hover { background:#dc2626; }
QPushButton#Success { background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2563eb, stop:1 #7c3aed); color:white; }
QPushButton#Success:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1d4ed8, stop:1 #6d28d9); }
QPushButton#Nav { background:transparent; color:#475569; border:1px solid transparent; text-align:left; padding:12px 14px; border-radius:13px; font-weight:800; }
QPushButton#Nav:hover { background:#eef4ff; color:#1d4ed8; }
QPushButton#Nav[active="true"] { background:#e0edff; color:#1d4ed8; border:1px solid #bfdbfe; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:#ffffff; color:#172033; border:1px solid #d7e0ec; border-radius:11px; padding:8px 10px; selection-background-color:#bfdbfe; }
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border:1px solid #3b82f6; background:#fbfdff; }
QComboBox::drop-down { border:0; width:28px; }
QCheckBox { color:#334155; spacing:8px; }
QCheckBox::indicator { width:18px; height:18px; border-radius:5px; border:1px solid #b8c6d8; background:#ffffff; }
QCheckBox::indicator:checked { background:#2563eb; border:1px solid #2563eb; }
QProgressBar { border:1px solid #d7e0ec; border-radius:12px; background:#edf2f7; height:25px; text-align:center; color:#334155; font-weight:800; }
QProgressBar::chunk { border-radius:11px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22c55e, stop:.55 #3b82f6, stop:1 #8b5cf6); }
QTableWidget { background:#ffffff; alternate-background-color:#f8fafc; color:#172033; gridline-color:#edf2f7; border:1px solid #dbe4f0; border-radius:13px; selection-background-color:#dbeafe; selection-color:#0f172a; }
QHeaderView::section { background:#f1f5f9; color:#334155; padding:9px; border:0; border-right:1px solid #e2e8f0; font-weight:850; }
QScrollBar:vertical { background:#f1f5f9; width:12px; margin:2px; border-radius:6px; }
QScrollBar::handle:vertical { background:#cbd5e1; border-radius:6px; min-height:32px; }
QScrollBar::handle:vertical:hover { background:#94a3b8; }
"""


def shadow(widget: QWidget, blur: int = 22, y: int = 8, alpha: int = 24) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setOffset(0, y)
    eff.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(eff)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--", note: str = ""):
        super().__init__()
        self.setObjectName("Card")
        shadow(self, 18, 7, 20)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        self.title = QLabel(title)
        self.title.setObjectName("Tiny")
        self.value = QLabel(value)
        self.value.setStyleSheet("color:#0f172a;font-size:27px;font-weight:950;")
        self.note = QLabel(note)
        self.note.setObjectName("Tiny")
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.note)

    def set_value(self, value: str) -> None:
        self.value.setText(value or "--")


class StepCard(QFrame):
    def __init__(self, name: str, desc: str):
        super().__init__()
        self.name = name
        self.setObjectName("Card")
        shadow(self, 14, 5, 18)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.badge = QLabel("待执行")
        self.badge.setFixedWidth(76)
        self.badge.setStyleSheet("color:#64748b;font-weight:900;")
        box = QVBoxLayout()
        title = QLabel(name)
        title.setStyleSheet("font-weight:900;color:#0f172a;")
        self.detail = QLabel(desc)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#64748b;font-size:12px;")
        box.addWidget(title)
        box.addWidget(self.detail)
        layout.addWidget(self.badge)
        layout.addLayout(box, 1)

    def reset(self) -> None:
        self.badge.setText("待执行")
        self.badge.setStyleSheet("color:#64748b;font-weight:900;")

    def running(self, command: str) -> None:
        self.badge.setText("运行中")
        self.badge.setStyleSheet("color:#2563eb;font-weight:950;")
        self.detail.setText(command)

    def done(self, ok: bool, code: int) -> None:
        if ok:
            self.badge.setText("完成")
            self.badge.setStyleSheet("color:#16a34a;font-weight:950;")
        else:
            self.badge.setText("失败")
            self.badge.setStyleSheet("color:#dc2626;font-weight:950;")
            self.detail.setText(self.detail.text() + f"  [returncode={code}]")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1540, 940)
        self.config = load_config()
        self.thread: WorkflowThread | None = None
        self.last_result: WorkflowResult | None = None
        self.cards: Dict[str, StepCard] = {}
        self.metrics: Dict[str, MetricCard] = {}
        self._build_ui()
        self._load_config()
        self._refresh_status()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QFrame(); sidebar.setObjectName("Sidebar"); sidebar.setFixedWidth(250)
        sl = QVBoxLayout(sidebar); sl.setContentsMargins(20, 24, 20, 20)
        logo = QLabel("SP4CPG\nHGCN Workbench")
        logo.setStyleSheet("font-size:22px;font-weight:950;color:#0f172a;line-height:125%;")
        sub = QLabel("Windows Client · HGCN + MEDE")
        sub.setStyleSheet("color:#2563eb;font-size:13px;font-weight:800;")
        sl.addWidget(logo); sl.addWidget(sub); sl.addSpacing(22)
        self.nav_buttons: List[QPushButton] = []
        for i, text in enumerate(["总览", "任务配置", "检测流程", "结果分析", "报告导出"]):
            b = QPushButton(text); b.setObjectName("Nav"); b.setMinimumHeight(45)
            b.clicked.connect(lambda checked=False, idx=i: self.pages.setCurrentIndex(idx))
            self.nav_buttons.append(b); sl.addWidget(b)
        sl.addStretch(1)
        self.repo_status = QLabel("未选择仓库")
        self.repo_status.setWordWrap(True)
        self.repo_status.setStyleSheet("color:#64748b;font-size:12px;background:#f8fafc;border:1px solid #dbe4f0;border-radius:14px;padding:12px;")
        sl.addWidget(self.repo_status)

        main = QWidget(); ml = QVBoxLayout(main); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)
        top = QFrame(); top.setObjectName("Topbar"); top.setFixedHeight(72)
        tl = QHBoxLayout(top); tl.setContentsMargins(26, 13, 26, 13)
        self.title = QLabel("HGCN + MEDE 漏洞检测工作台"); self.title.setObjectName("Title")
        tl.addWidget(self.title); tl.addStretch(1)
        self.start_btn = QPushButton("启动任务"); self.start_btn.setObjectName("Success"); self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("停止"); self.stop_btn.setObjectName("Danger"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        tl.addWidget(self.start_btn); tl.addWidget(self.stop_btn)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._overview_page())
        self.pages.addWidget(self._config_page())
        self.pages.addWidget(self._workflow_page())
        self.pages.addWidget(self._results_page())
        self.pages.addWidget(self._export_page())
        self.pages.currentChanged.connect(self._page_changed)
        ml.addWidget(top); ml.addWidget(self.pages, 1)
        root_layout.addWidget(sidebar); root_layout.addWidget(main, 1)
        self._page_changed(0)

    def _overview_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(28,28,28,28)
        hero = QFrame(); hero.setObjectName("Hero"); shadow(hero, 26, 10, 26)
        h = QVBoxLayout(hero); h.setContentsMargins(26,24,26,24)
        t = QLabel("HGCN 深度检测 + MEDE 静态证据验证")
        t.setObjectName("HeroTitle")
        d = QLabel("本客户端将深度检测核心切换为 HGCN，并把 MEDE 静态指针分析纳入同一条流水线。支持生成 CPG/HCPG 图工件、加载已训练 HGCN 模型检测、运行 MEDE 静态指针分析、交叉验证以及 HTML/CSV/JSON 报告导出。")
        d.setObjectName("HeroText"); d.setWordWrap(True)
        h.addWidget(t); h.addWidget(d)
        layout.addWidget(hero)
        grid = QGridLayout(); grid.setSpacing(16)
        items = [("Accuracy", "--"), ("Precision", "--"), ("Recall", "--"), ("F1", "--"), ("高风险告警", "--"), ("检测模式", "--")]
        for i, (name, val) in enumerate(items):
            card = MetricCard(name, val)
            self.metrics[name] = card
            grid.addWidget(card, i // 3, i % 3)
        layout.addLayout(grid)
        info = QGroupBox("当前配置")
        g = QGridLayout(info)
        self.cur_repo = QLabel("--"); self.cur_repo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cur_dataset = QLabel("--"); self.cur_model = QLabel("--"); self.cur_MEDE = QLabel("--")
        for i, (k, v) in enumerate([("仓库", self.cur_repo), ("数据集", self.cur_dataset), ("HGCN", self.cur_model), ("MEDE", self.cur_MEDE)]):
            g.addWidget(QLabel(k), i, 0); g.addWidget(v, i, 1)
        layout.addWidget(info); layout.addStretch(1)
        return page

    def _config_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(28,28,28,28)
        pbox = QGroupBox("项目与数据")
        g = QGridLayout(pbox)
        self.repo_edit = QLineEdit(); self.py_edit = QLineEdit(); self.dataset_edit = QLineEdit(); self.out_edit = QLineEdit()
        self._row(g, 0, "仓库根目录", self.repo_edit, True)
        self._row(g, 1, "Python 解释器", self.py_edit, False)
        self._row(g, 2, "数据集文件(data/)或绝对路径", self.dataset_edit, False)
        self._row(g, 3, "输出目录", self.out_edit, True)
        layout.addWidget(pbox)

        cbox = QGroupBox("HGCN 深度检测")
        cg = QGridLayout(cbox)
        self.mode_combo = QComboBox(); self.mode_combo.addItems(["完整流程：HGCN + MEDE + 交叉验证", "使用已训练 HGCN 检测", "重新训练 HGCN"] )
        self.model_edit = QLineEdit(); self.ckpt_edit = QLineEdit()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 1000)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 256)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0000001, 1.0); self.lr_spin.setDecimals(8); self.lr_spin.setSingleStep(0.00001)
        self.maxlen_spin = QSpinBox(); self.maxlen_spin.setRange(1, 1000); self.maxlen_spin.setSingleStep(10)
        self.fallback_cb = QCheckBox("允许 fallback 保底跑通（缺少 HGCN 权重或 MEDE 时生成标记为 fallback 的结果）")
        cg.addWidget(QLabel("任务模式"), 0, 0); cg.addWidget(self.mode_combo, 0, 1, 1, 3)
        cg.addWidget(QLabel("HGCN 模型类型"), 1, 0); cg.addWidget(self.model_edit, 1, 1, 1, 3)
        cg.addWidget(QLabel("已训练模型文件(.pt)"), 2, 0); cg.addWidget(self.ckpt_edit, 2, 1, 1, 2); cg.addWidget(self._browse_btn(self.ckpt_edit, False), 2, 3)
        for col, (label, widget) in enumerate([("epoch", self.epochs_spin), ("batch", self.batch_spin), ("lr", self.lr_spin), ("patience", self.maxlen_spin)]):
            cg.addWidget(QLabel(label), 3, col, Qt.AlignLeft)
            cg.addWidget(widget, 4, col)
        cg.addWidget(self.fallback_cb, 5, 0, 1, 4)
        self.split_edit = QLineEdit()
        cg.addWidget(QLabel("训练划分清单 JSON"), 6, 0)
        cg.addWidget(self.split_edit, 6, 1, 1, 2)
        cg.addWidget(self._browse_btn(self.split_edit, False), 6, 3)
        self.model_edit.setPlaceholderText("TAF-HGCN / HGNN / HGCN / GCN / GAT / GIN / GraphSAGE / GGNN")
        layout.addWidget(cbox)

        sbox = QGroupBox("MEDE 静态指针分析")
        sg = QGridLayout(sbox)
        self.enable_MEDE_cb = QCheckBox("启用 MEDE/静态证据检测")
        self.MEDE_home_edit = QLineEdit(); self.bc_edit = QLineEdit(); self.MEDE_cmd_edit = QLineEdit(); self.static_edit = QLineEdit()
        sg.addWidget(self.enable_MEDE_cb, 0, 0, 1, 4)
        self._row(sg, 1, "MEDE_HOME", self.MEDE_home_edit, True)
        self._row(sg, 2, "LLVM bitcode(.bc/.ll)", self.bc_edit, False)
        sg.addWidget(QLabel("自定义 MEDE 命令"), 3, 0); sg.addWidget(self.MEDE_cmd_edit, 3, 1, 1, 3)
        self._row(sg, 4, "已有静态结果 JSON", self.static_edit, False)
        layout.addWidget(sbox)

        abox = QGroupBox("高级选项：原 HCPG 工件和语义配置")
        ag = QGridLayout(abox)
        self.run_cpg_cb = QCheckBox("生成 CPG DOT")
        self.run_hcpg_cb = QCheckBox("生成 HCPG DOT")
        self.run_embed_cb = QCheckBox("生成 HCPG embedding")
        self.joern_edit = QLineEdit(); self.skills_edit = QLineEdit(); self.llm_edit = QLineEdit()
        ag.addWidget(self.run_cpg_cb, 0, 0); ag.addWidget(self.run_hcpg_cb, 0, 1); ag.addWidget(self.run_embed_cb, 0, 2)
        self._row(ag, 1, "JOERN_HOME", self.joern_edit, True)
        self._row(ag, 2, "SKILLs 目录", self.skills_edit, True)
        self._row(ag, 3, "LLM 配置文件", self.llm_edit, False)
        layout.addWidget(abox)

        buttons = QHBoxLayout()
        save = QPushButton("保存配置"); save.clicked.connect(self.save_config)
        check = QPushButton("环境检查"); check.setObjectName("Secondary"); check.clicked.connect(self.check_env)
        buttons.addWidget(save); buttons.addWidget(check); buttons.addStretch(1)
        layout.addLayout(buttons); layout.addStretch(1)
        return page

    def _workflow_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(28,28,28,28)
        box = QGroupBox("检测流水线")
        v = QVBoxLayout(box)
        definitions = [
            ("可选：CPG 图生成", "调用原仓库 preprocess/cpg_generate.py，生成 CPG DOT 工件。"),
            ("可选：HCPG 图构建", "调用 hcpg_generate.py，生成 HCPG DOT 工件。"),
            ("可选：HCPG 嵌入生成", "调用 dot_embedding.py，生成图嵌入数据。"),
            ("HGCN 重新训练与评估", "调用原仓库 main.py --model HGCN 重新训练图模型并读取训练日志指标。"),
            ("HGCN 漏洞检测", "加载已训练 HGCN .pt 模型，对 HCPG 数据集执行函数级漏洞预测。"),
            ("MEDE 静态指针分析", "调用 MEDE wpa 或内置静态扫描，生成可用于交叉验证的静态证据。"),
            ("交叉验证与综合判定", "融合 HGCN 检测结果和 MEDE/静态证据，生成高/中/低风险告警。"),
        ]
        for name, desc in definitions:
            card = StepCard(name, desc); self.cards[name] = card; v.addWidget(card)
        self.progress = QProgressBar(); self.progress.setValue(0)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(260)
        v.addWidget(self.progress); v.addWidget(self.log, 1)
        layout.addWidget(box, 1)
        return page

    def _results_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(28,28,28,28)
        summary = QGroupBox("结果摘要")
        g = QGridLayout(summary)
        self.summary_labels: Dict[str, QLabel] = {}
        for i, key in enumerate(["状态", "输出目录", "HGCN预测", "MEDE静态结果", "交叉验证", "模型目录"]):
            lab = QLabel("--"); lab.setTextInteractionFlags(Qt.TextSelectableByMouse); self.summary_labels[key] = lab
            g.addWidget(QLabel(key), i//2, (i%2)*2); g.addWidget(lab, i//2, (i%2)*2+1)
        layout.addWidget(summary)
        table_box = QGroupBox("告警列表 / 预测结果")
        tb = QVBoxLayout(table_box)
        self.result_table = QTableWidget(); self.result_table.setAlternatingRowColors(True)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tb.addWidget(self.result_table)
        layout.addWidget(table_box, 1)
        return page

    def _export_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(28,28,28,28)
        box = QGroupBox("报告导出")
        v = QVBoxLayout(box)
        info = QLabel("检测流程会自动保存 JSON/CSV 文件。这里可额外导出 HTML 报告，便于验收、审计和展示。")
        info.setWordWrap(True); info.setObjectName("Muted")
        v.addWidget(info)
        buttons = QHBoxLayout()
        html = QPushButton("导出 HTML 报告"); html.clicked.connect(self.export_html)
        out = QPushButton("打开输出目录"); out.setObjectName("Secondary"); out.clicked.connect(self.open_output_dir)
        buttons.addWidget(html); buttons.addWidget(out); buttons.addStretch(1)
        v.addLayout(buttons)
        self.export_log = QTextEdit(); self.export_log.setReadOnly(True)
        v.addWidget(self.export_log, 1)
        layout.addWidget(box, 1)
        return page

    def _row(self, grid: QGridLayout, row: int, label: str, edit: QLineEdit, directory: bool) -> None:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(edit, row, 1, 1, 2)
        grid.addWidget(self._browse_btn(edit, directory), row, 3)

    def _browse_btn(self, target: QLineEdit, directory: bool) -> QPushButton:
        btn = QPushButton("浏览"); btn.setObjectName("Secondary")
        def browse():
            base = target.text() or str(Path.home())
            if directory:
                path = QFileDialog.getExistingDirectory(self, "选择目录", base)
            else:
                path, _ = QFileDialog.getOpenFileName(self, "选择文件", base, "All Files (*.*)")
            if path: target.setText(path)
        btn.clicked.connect(browse)
        return btn

    def _load_config(self) -> None:
        c = self.config
        self.repo_edit.setText(c.repo_root); self.py_edit.setText(c.python_exe); self.dataset_edit.setText(c.dataset_name); self.out_edit.setText(c.output_dir)
        mode_map = {"full": 0, "predict": 1, "train": 2}
        self.mode_combo.setCurrentIndex(mode_map.get(c.task_mode, 0))
        self.model_edit.setText(c.hgcn_model_name); self.ckpt_edit.setText(c.hgcn_checkpoint)
        self.split_edit.setText(c.hgcn_split_manifest)
        self.epochs_spin.setValue(c.hgcn_epochs); self.batch_spin.setValue(c.hgcn_batch); self.lr_spin.setValue(c.hgcn_lr); self.maxlen_spin.setValue(c.hgcn_patience)
        self.fallback_cb.setChecked(c.allow_fallback)
        self.enable_MEDE_cb.setChecked(c.enable_MEDE); self.MEDE_home_edit.setText(c.MEDE_home); self.bc_edit.setText(c.bitcode_path); self.MEDE_cmd_edit.setText(c.MEDE_command); self.static_edit.setText(c.static_result_path)
        self.run_cpg_cb.setChecked(c.run_cpg); self.run_hcpg_cb.setChecked(c.run_hcpg); self.run_embed_cb.setChecked(c.run_embedding)
        self.joern_edit.setText(c.joern_home); self.skills_edit.setText(c.skills_home); self.llm_edit.setText(c.llm_config_path)

    def _config_from_ui(self) -> ClientConfig:
        mode_index = self.mode_combo.currentIndex()
        mode = "full" if mode_index == 0 else "predict" if mode_index == 1 else "train"
        return ClientConfig(
            repo_root=self.repo_edit.text().strip(), python_exe=self.py_edit.text().strip() or "python",
            dataset_name=self.dataset_edit.text().strip() or "function.json", output_dir=self.out_edit.text().strip(),
            task_mode=mode, hgcn_model_name=self.model_edit.text().strip() or "HGCN",
            hgcn_checkpoint=self.ckpt_edit.text().strip(), hgcn_epochs=self.epochs_spin.value(),
            hgcn_split_manifest=self.split_edit.text().strip(),
            hgcn_batch=self.batch_spin.value(), hgcn_lr=self.lr_spin.value(), hgcn_patience=self.maxlen_spin.value(),
            allow_fallback=self.fallback_cb.isChecked(), run_cpg=self.run_cpg_cb.isChecked(), run_hcpg=self.run_hcpg_cb.isChecked(), run_embedding=self.run_embed_cb.isChecked(),
            enable_MEDE=self.enable_MEDE_cb.isChecked(), MEDE_home=self.MEDE_home_edit.text().strip(), bitcode_path=self.bc_edit.text().strip(), MEDE_command=self.MEDE_cmd_edit.text().strip(), static_result_path=self.static_edit.text().strip(),
            llm_config_path=self.llm_edit.text().strip(), skills_home=self.skills_edit.text().strip(), joern_home=self.joern_edit.text().strip()
        )

    def _page_changed(self, idx: int) -> None:
        titles = ["总览", "任务配置", "检测流程", "结果分析", "报告导出"]
        self.title.setText("HGCN + MEDE 漏洞检测工作台 · " + titles[idx])
        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", "true" if i == idx else "false")
            btn.style().unpolish(btn); btn.style().polish(btn)
        self._refresh_status()

    def _refresh_status(self) -> None:
        c = self._config_from_ui()
        repo = Path(c.repo_root) if c.repo_root else Path()
        data = c.data_path
        ok = bool(c.repo_root and (repo / "data").exists())
        self.repo_status.setText(("仓库已就绪\n" if ok else "未检测到有效仓库\n") + str(repo if c.repo_root else "请选择仓库根目录"))
        self.cur_repo.setText(c.repo_root or "--")
        self.cur_dataset.setText(str(data))
        self.cur_model.setText(c.hgcn_model_name)
        self.cur_MEDE.setText("启用" if c.enable_MEDE else "关闭")
        self.metrics["检测模式"].set_value(self.mode_combo.currentText().split("：")[0])

    def save_config(self) -> None:
        self.config = self._config_from_ui(); save_config(self.config); self._refresh_status()
        QMessageBox.information(self, "保存成功", "配置已保存。")

    def check_env(self) -> None:
        c = self._config_from_ui(); problems = []
        if not c.repo_root: problems.append("未选择仓库根目录")
        elif not (c.repo / "data").exists(): problems.append("仓库下缺少 data/ 目录")
        if not c.data_path.exists(): problems.append(f"数据集不存在：{c.data_path}")
        if c.task_mode in {"predict", "full"} and not c.hgcn_checkpoint and not c.allow_fallback:
            problems.append("未提供 HGCN checkpoint，且 fallback 未启用")
        if c.task_mode == "train" and not Path(c.hgcn_split_manifest).is_file():
            problems.append("训练需要有效的样本划分清单 JSON")
        if c.enable_MEDE and not (c.MEDE_command or c.static_result_path or c.bitcode_path or c.allow_fallback):
            problems.append("MEDE 未配置 bitcode/命令/已有结果，且 fallback 未启用")
        if c.run_cpg and not (c.preprocess_dir / "cpg_generate.py").exists(): problems.append("启用 CPG 但缺少 preprocess/cpg_generate.py")
        if problems:
            QMessageBox.warning(self, "环境检查", "发现问题：\n" + "\n".join(problems))
        else:
            QMessageBox.information(self, "环境检查", "基础配置检查通过。真实 HGCN/MEDE 依赖会在任务启动时由子进程验证；若启用 fallback，仅用于流程演示，结果会明确标记。")

    def start(self) -> None:
        self.config = self._config_from_ui(); save_config(self.config); self._refresh_status()
        for c in self.cards.values(): c.reset()
        self.log.clear(); self.progress.setValue(0)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.thread = WorkflowThread(self.config, self.config.task_mode)
        s = self.thread.streamer
        s.line.connect(self.append_log); s.step_started.connect(self.step_started); s.step_finished.connect(self.step_finished)
        s.progress.connect(self.progress.setValue); s.finished.connect(self.finished); s.failed.connect(self.failed)
        s.finished.connect(self.thread.quit); s.failed.connect(self.thread.quit)
        self.thread.start(); self.pages.setCurrentIndex(2)

    def stop(self) -> None:
        if self.thread: self.thread.stop(); self.append_log("[client] 正在请求停止任务...")

    def append_log(self, text: str) -> None:
        self.log.append(text); self.log.moveCursor(QTextCursor.End)

    def step_started(self, name: str, command: str) -> None:
        if name in self.cards: self.cards[name].running(command)

    def step_finished(self, name: str, ok: bool, code: int) -> None:
        if name in self.cards: self.cards[name].done(ok, code)

    def finished(self, result: WorkflowResult) -> None:
        self.last_result = result; self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.append_log("\n[client] " + result.message)
        self.update_results(result); self.pages.setCurrentIndex(3)

    def failed(self, msg: str) -> None:
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.append_log("[client:error] " + msg); QMessageBox.critical(self, "任务失败", msg)

    def update_results(self, r: WorkflowResult) -> None:
        self.summary_labels["状态"].setText("成功" if r.ok else "失败")
        self.summary_labels["输出目录"].setText(r.output_dir)
        self.summary_labels["HGCN预测"].setText(r.artifacts.get("模型预测 JSON", "--"))
        self.summary_labels["MEDE静态结果"].setText(r.artifacts.get("MEDE 静态结果 JSON", "--"))
        self.summary_labels["交叉验证"].setText(r.artifacts.get("综合报告 JSON", "--"))
        self.summary_labels["模型目录"].setText(r.artifacts.get("HGCN 模型目录", "--"))
        self.metrics["Accuracy"].set_value(r.metrics.get("accuracy", r.metrics.get("Test Accuracy", "--")))
        self.metrics["Precision"].set_value(r.metrics.get("precision", r.metrics.get("Precision", "--")))
        self.metrics["Recall"].set_value(r.metrics.get("recall", r.metrics.get("Recall", "--")))
        self.metrics["F1"].set_value(r.metrics.get("f1", r.metrics.get("F1 Score", "--")))
        self.metrics["高风险告警"].set_value(self._count_high(r.result_table_path))
        self._load_table(r.result_table_path)

    def _count_high(self, csv_path: str) -> str:
        if not csv_path or not Path(csv_path).exists(): return "--"
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            return str(sum(1 for row in rows if str(row.get("risk", "")).lower() == "high" or str(row.get("validation", "")).lower() == "confirmed"))
        except Exception:
            return "--"

    def _load_table(self, csv_path: str) -> None:
        self.result_table.clear()
        if not csv_path or not Path(csv_path).exists():
            self.result_table.setRowCount(0); self.result_table.setColumnCount(0); return
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f); rows = list(reader); headers = reader.fieldnames or []
        self.result_table.setColumnCount(len(headers)); self.result_table.setHorizontalHeaderLabels(headers); self.result_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, h in enumerate(headers):
                self.result_table.setItem(r, c, QTableWidgetItem(str(row.get(h, ""))))

    def export_html(self) -> None:
        if not self.last_result:
            QMessageBox.warning(self, "没有结果", "请先运行一次检测任务。")
            return
        out = Path(self.last_result.output_dir) / "hgcn_MEDE_client_report.html"
        export_html_report(self.last_result, out)
        self.export_log.append(f"HTML 报告已导出：{out}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def open_output_dir(self) -> None:
        out = self._config_from_ui().resolved_output_dir; out.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(QSS)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
