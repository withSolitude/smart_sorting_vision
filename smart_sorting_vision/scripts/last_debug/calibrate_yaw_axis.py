"""标定顶部视觉 Yaw 与机器人角度关系；统一从 last_debug 入口运行。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/last_use/calibration/calibrate_top_yaw.py")
