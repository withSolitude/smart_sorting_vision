"""采集顶部 YOLO 分割数据；统一从 last_debug 入口运行。

启动前先加载正式 YAML，确保采集和运行使用同一套内参、ROI、曝光和分辨率。
"""
try:
    from ._delegate import run
    from .common import load_context
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run
    from scripts.last_debug.common import load_context


if __name__ == "__main__":
    load_context()
    run("scripts/dataset/top_seg/capture_top_dataset.py")
