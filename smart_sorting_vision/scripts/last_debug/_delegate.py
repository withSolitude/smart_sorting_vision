"""last_debug 统一入口的内部转发工具。

调试手册只暴露 ``scripts.last_debug`` 下的命令。这里复用已经验证过的
旧流程实现，避免复制一份会逐渐失配的相机、通信和标定逻辑。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(relative_path: str) -> None:
    """在工程根目录下运行一个现有调试脚本，并原样转发命令行参数。"""
    target = (ROOT / relative_path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"找不到调试脚本：{target}")
    # 让目标脚本看到自己的文件名，argparse 的帮助和错误信息更清楚。
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
