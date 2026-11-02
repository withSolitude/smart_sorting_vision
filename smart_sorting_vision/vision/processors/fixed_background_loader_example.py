from pathlib import Path

from vision.processors.side_object_extractor import (
    SideObjectExtractor,
)


ROOT = Path(__file__).resolve().parents[1]

BACKGROUND_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "side_camera"
    / "side_mechanical_roi_background.png"
)


extractor = SideObjectExtractor(
    padding_ratio=0.15
)

# 正式运行时只做这一句：
extractor.load_background(
    BACKGROUND_PATH
)

# 之后直接：
#
# mechanical_roi = camera.read()
# object_result = extractor.extract(
#     mechanical_roi
# )
