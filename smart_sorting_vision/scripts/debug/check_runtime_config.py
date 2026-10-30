from __future__ import annotations

from pathlib import Path

from runtime.app_config import load_runtime_config


# 当前文件位于 smart_sorting_vision/scripts/debug/，项目根是 parents[2]。
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "runtime_config.yaml"


def main() -> None:
    cfg = load_runtime_config(ROOT, CONFIG)

    print("=" * 72)
    print("SMART SORTING RUNTIME CONFIG CHECK")
    print("=" * 72)
    print(f"project : {ROOT}")
    print(f"config  : {CONFIG}")

    print("\n[CAMERA MAPPING]")
    for name in ("top", "side"):
        device = Path(cfg["camera"][name]["device"])
        status = "OK" if device.exists() else "MISSING"
        print(f"{name:>4} : {device}  [{status}]")

    print("\n[RESOURCE FILES]")
    resources = {
        "top model": cfg["model"]["top"]["path"],
        "side model": cfg["model"]["side"]["path"],
        "top intrinsics": cfg["calibration"]["top_intrinsics"],
        "pixel_to_robot": cfg["calibration"]["pixel_to_robot"],
        "side background": cfg["calibration"]["side_background"],
    }
    missing = []
    for name, value in resources.items():
        path = Path(value)
        status = "OK" if path.exists() else "MISSING"
        print(f"{name:>16} : {path}  [{status}]")
        if not path.exists():
            missing.append(str(path))

    x_limits = cfg["mechanics"]["workspace_mm"]["x"]
    y_limits = cfg["mechanics"]["workspace_mm"]["y"]
    print("\n[MECHANICAL WORKSPACE]")
    print(f"X: {x_limits[0]} .. {x_limits[1]} mm")
    print(f"Y: {y_limits[0]} .. {y_limits[1]} mm")

    top_c = cfg["vision"]["top"]["consensus"]
    side_c = cfg["vision"]["side"]["consensus"]
    print("\n[CONSENSUS]")
    print(f"top : {top_c['required']}/{top_c['window']}")
    print(f"side: {side_c['required']}/{side_c['window']}")

    if missing:
        raise SystemExit("\nFAIL: production resource files are missing")

    print("\nPASS: config syntax and production resource paths are valid.")
    print("NOTE: camera entries marked MISSING must be corrected before main.py.")


if __name__ == "__main__":
    main()
