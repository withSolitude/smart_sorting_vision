"""用实际抓取物验证 Yaw 对齐；统一从 last_debug 入口运行。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/last_use/validation/calibrate_real_grasp_yaw_alignment.py")
