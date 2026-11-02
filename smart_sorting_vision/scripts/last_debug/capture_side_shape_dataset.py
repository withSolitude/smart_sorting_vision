"""侧面分类数据集采集入口。

启动后按实际保存目录逐层选择：
数据集版本/train/val/test -> 形状 -> 颜色（如果该目录存在）。
"""

try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    # 目标脚本只扫描当前目录的直接子目录，并把图像写入最终选中的目录。
    run("scripts/last_use/dataset/capture_side_cls_dataset.py")
