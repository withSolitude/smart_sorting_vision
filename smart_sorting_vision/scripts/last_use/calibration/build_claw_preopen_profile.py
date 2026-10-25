"""Offline only: measured servo_deg:inner_gap_mm -> paired PC/MCU profiles.

Does not access any serial port, move motors, or change runtime_config.yaml.
"""
from __future__ import annotations
import argparse
import yaml
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vision.grasp.claw_preopen_profile import ClawPreopenProfile, firmware_header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", action="append", required=True,
                        help="measured claw_servo_deg:inner_gap_mm; repeat 3..32 times (NOT yaw)")
    parser.add_argument("--margin-mm", type=float, default=3.0)
    parser.add_argument("--confirmed-safe", action="store_true",
                        help="confirm all supplied positions were measured, collision-free PREOPEN positions, not final force-close")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.confirmed_safe:
        parser.error("measure real safe positions first; then provide --confirmed-safe")
    samples = []
    try:
        for text in args.point:
            angle, gap = (float(v) for v in text.split(":"))
            samples.append({"servo_deg": angle, "gap_mm": gap})
        data = {"schema_version": 1, "calibrated": True,
                "created_at": datetime.now().astimezone().isoformat(),
                "source": "user measured inner gaps at fixed contact height",
                "margin_mm": args.margin_mm, "samples": samples}
        profile = ClawPreopenProfile(data)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    out = args.out_dir or ROOT / "outputs" / "claw_preopen" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    pc = out / "dynamic_preopen.yaml"
    mcu = out / "claw_preopen_profile.h"
    runtime = {
        "enabled": True, "configured": True, "profile_kind": "measured",
        "hardware_validated": True, "closed_servo_deg": profile.points[0][1],
        "standby_servo_deg": profile.points[-1][1],
        "margin_mm": args.margin_mm, "min_command_mm": profile.min_command_mm,
        "max_command_mm": profile.max_command_mm,
    }
    pc.write_text(yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding="utf-8")
    mcu.write_text(firmware_header(data), encoding="utf-8")
    print(f"YAML block: {pc}\nMCU header: {mcu}")
    print(f"Allowed preopen: {profile.min_command_mm:.1f} .. {profile.max_command_mm:.1f} mm")
    print(f"Maximum object width with {profile.margin_mm:g} mm TOTAL margin: "
          f"{profile.max_command_mm-profile.margin_mm:.1f} mm")
    print("Install BOTH outputs, rebuild and flash MCU. Files are NOT installed automatically.")
    print("Test several intermediate openings with calipers before automatic picking.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
