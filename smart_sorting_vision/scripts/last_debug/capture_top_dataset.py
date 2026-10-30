"""顶部数据集采集入口。

目录菜单直接扫描实际数据集目录，只显示当前层级的直接子目录：
train/val/test -> 场景目录 -> 颜色/形状等下一级目录。
"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    # 复用已经按目标目录逐层选择的采集实现；它自己读取正式 YAML，
    # 因此不会在 last_debug 入口维护第二套相机参数。
    run("scripts/last_use/dataset/capture_top_seg_dataset.py")
