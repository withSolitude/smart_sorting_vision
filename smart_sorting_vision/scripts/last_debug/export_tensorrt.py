"""在最终 Jetson 上把已验证的 YOLO 权重导出为 TensorRT engine。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/last_use/export_tensorrt.py")
