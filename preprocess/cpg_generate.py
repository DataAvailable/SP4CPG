import os
import json
import shutil
import subprocess
from pathlib import Path

# 路径配置
json_file = "../data/function.json"  # 原始数据集文件路径
dot_output_dir = Path("dots")
joern_workspace = Path("workspace")

# 创建目录
dot_output_dir.mkdir(parents=True, exist_ok=True)
joern_workspace.mkdir(parents=True, exist_ok=True)

# 读取 JSON 文件
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, item in enumerate(data):
    code = item["func"]
    label = item["target"]

    # 保存为 .c 文件
    c_source_dir = Path("source/" + f"{idx}")
    c_source_dir.mkdir(parents=True, exist_ok=True)
    c_file = c_source_dir / f"{label}.c"
    with open(c_file, "w", encoding="utf-8") as cf:
        cf.write(code)

    # 使用 Joern 生成 CPG
    cpg_bin_path = joern_workspace / f"{idx}-cpg-{label}.bin"
    dot_path = dot_output_dir / f"{idx}"
    joern_cmds = [
        f"joern-parse {c_source_dir} -o {cpg_bin_path}",
        f"joern-export {cpg_bin_path} --repr cpg14 --out {dot_path}"
    ]

    print(f"[+] Processing: {c_file}")
    for cmd in joern_cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] Error: {cmd}")
            print(result.stderr)
            continue

    # 获取并重命名 dot 文件
    generated_dot = dot_path / "0-cpg.dot"
    if generated_dot.exists():
        renamed_dot = dot_output_dir / f"{idx}-cpg-{label}.dot"
        generated_dot.rename(renamed_dot)
        print(f"    --> Saved to {renamed_dot}")
    else:
        print(f"[!] Dot file not found for {c_file.name}")

    if os.path.exists(dot_path) and os.path.isdir(dot_path):
        shutil.rmtree(dot_path)
    else:
        print(f"错误: 文件夹 '{dot_path}' 不存在")
    # 清理 Joern 输出目录
    # for f in (dot_path).glob("*.dot"):
    #     f.unlink()

print("\n✅ All files processed.")