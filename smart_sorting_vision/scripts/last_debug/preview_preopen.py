"""离线预览预开口角度/间隙关系；不连接电机。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/last_use/calibration/preview_claw_preopen.py")
