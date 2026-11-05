"""
ObjectInfo

定义侧面视觉识别后的物体信息。
当前侧面流水线只能可靠给出“二维码是否存在”，因此这里使用
qr_present: bool，而不是伪造二维码内容字符串。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectInfo:
    """单个物体的完整分箱属性。"""

    color: str
    shape: str
    qr_present: bool = False

    def key(self) -> str:
        """生成稳定、无歧义的分箱键。"""
        return (
            f"{self.color}|"
            f"{self.shape}|"
            f"qr_present={int(bool(self.qr_present))}"
        )
