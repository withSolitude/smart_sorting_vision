"""
BoxManager

负责将侧面识别结果:
color / qr / shape
转换为:
box_id (1~6)
"""

from dataclasses import dataclass


@dataclass
class BoxResult:
    success: bool
    box_id: int
    reason: str


class BoxManager:

    def __init__(self):

        self.rules = [
            {
                "box_id": 1,
                "color": "red",
                "shape": None,
                "qr": None,
            },
            {
                "box_id": 2,
                "color": "blue",
                "shape": None,
                "qr": None,
            },
            {
                "box_id": 3,
                "color": None,
                "shape": "三棱柱",
                "qr": None,
            },
            {
                "box_id": 4,
                "color": None,
                "shape": "正方体",
                "qr": None,
            },
            {
                "box_id": 5,
                "color": None,
                "shape": None,
                "qr": "A",
            },
            {
                "box_id": 6,
                "color": None,
                "shape": None,
                "qr": "B",
            },
        ]


    def classify(
        self,
        color=None,
        shape=None,
        qr=None,
    ):

        for rule in self.rules:

            if rule["color"] is not None and rule["color"] != color:
                continue

            if rule["shape"] is not None and rule["shape"] != shape:
                continue

            if rule["qr"] is not None and rule["qr"] != qr:
                continue

            return BoxResult(
                True,
                rule["box_id"],
                "matched"
            )

        return BoxResult(
            False,
            0,
            "no_matching_rule"
        )
