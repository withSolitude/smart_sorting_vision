from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MAPPING_PATH = (
    ROOT / "data" / "calibration" / "top_camera" / "robot_mapping" / "pixel_to_robot.json"
)
REPORT_PATH = (
    ROOT / "data" / "calibration" / "top_camera" / "robot_mapping" / "tray8_validation_report.json"
)

MIN_POINTS = 5
MAX_MEAN_MM = 2.0
MAX_P95_MM = 3.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MAPPING_PATH.is_file():
        raise FileNotFoundError(f"mapping不存在: {MAPPING_PATH}")
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"独立验证报告不存在: {REPORT_PATH}\n"
            "先运行 verify_top_robot_tray8.py 并至少采5个未参与拟合的验证点。"
        )

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    current_hash = sha256(MAPPING_PATH)
    report_hash = str(report.get("calibration_sha256", ""))
    if report_hash != current_hash:
        raise RuntimeError(
            "验证报告不是针对当前pixel_to_robot.json生成的，禁止提升为正式标定。\n"
            f"current={current_hash}\nreport ={report_hash or '<missing>'}"
        )

    points = int(report.get("point_count", 0))
    mean_mm = float(report.get("mean_error_mm", float("inf")))
    p95_mm = float(report.get("p95_error_mm", float("inf")))
    if points < MIN_POINTS or mean_mm > MAX_MEAN_MM or p95_mm > MAX_P95_MM:
        raise RuntimeError(
            "独立验证未达到当前工程门槛，禁止提升为正式标定: "
            f"points={points} (>= {MIN_POINTS}), "
            f"mean={mean_mm:.3f}mm (<= {MAX_MEAN_MM}), "
            f"P95={p95_mm:.3f}mm (<= {MAX_P95_MM})"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MAPPING_PATH.with_name(f"pixel_to_robot_preverify_{stamp}.json")
    shutil.copy2(MAPPING_PATH, backup)

    metadata = mapping.setdefault("metadata", {})
    metadata["verification_status"] = "verified"
    metadata["verified_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["validated_mapping_sha256"] = current_hash
    metadata["independent_validation"] = {
        "point_count": points,
        "mean_error_mm": mean_mm,
        "rmse_mm": float(report.get("rmse_mm", float("nan"))),
        "p95_error_mm": p95_mm,
        "max_error_mm": float(report.get("max_error_mm", float("nan"))),
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }

    temp = MAPPING_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(MAPPING_PATH)

    print("[PASS] pixel_to_robot.json 已标记为 verified")
    print(f"backup: {backup}")
    print(f"points={points}, mean={mean_mm:.3f} mm, P95={p95_mm:.3f} mm")


if __name__ == "__main__":
    main()
