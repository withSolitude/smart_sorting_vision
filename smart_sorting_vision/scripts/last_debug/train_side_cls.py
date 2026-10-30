"""训练侧面形状/no_object 分类模型；统一从 last_debug 入口运行。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/train/side_cls/train_side_cls.py")
