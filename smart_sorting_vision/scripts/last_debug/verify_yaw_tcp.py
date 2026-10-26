"""验证 Yaw 变化时夹爪 TCP 中心是否漂移；统一从 last_debug 入口运行。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/last_use/calibration/verify_top_yaw_tcp_compensated.py")
