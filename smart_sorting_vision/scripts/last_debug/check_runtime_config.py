"""检查 runtime_config.yaml 是否能被工程加载。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/runtime_config.yaml",
        help="相对工程根目录的 YAML 路径，默认使用正式配置",
    )
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime.app_config import load_runtime_config

    path = Path(args.config).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    cfg = load_runtime_config(ROOT, path.resolve())
    print(f"config ok: {path.resolve()}")
    print(f"top camera: {cfg['camera']['top']['width']}x{cfg['camera']['top']['height']}")
    print(f"side camera: {cfg['camera']['side']['width']}x{cfg['camera']['side']['height']}")
    print(f"top intrinsics: {cfg['calibration']['top_intrinsics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
