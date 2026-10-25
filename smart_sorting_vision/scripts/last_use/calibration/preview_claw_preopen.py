"""Print the installed CAD preopen mapping; offline, no motion or serial port."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from vision.grasp.claw_preopen_profile import ClawPreopenProfile

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=float, help='Detected object width in mm; total margin is added')
    parser.add_argument('--gap', type=float, help='Requested net opening in mm; no extra margin')
    args = parser.parse_args()
    runtime = yaml.safe_load((ROOT / "config" / "runtime_config.yaml").read_text(encoding="utf-8"))
    cfg = runtime["vision"]["top"]["grasp_orientation"]["dynamic_preopen"]
    profile = ClawPreopenProfile.from_runtime_config(cfg)
    print(f'Profile={profile.kind}; command={profile.min_command_mm:.1f}..{profile.max_command_mm:.1f} mm; total margin={profile.margin_mm:g} mm')
    print('CLAW servo degrees, not Yaw. No hardware is connected by this command.')
    if args.width is not None:
        gap = profile.command_for_width(args.width)
        print(f'Object={args.width:g} mm -> command gap={gap:.1f} mm (servo conversion occurs in MCU)')
    elif args.gap is not None:
        gap = float(args.gap)
        if not profile.min_command_mm <= gap <= profile.max_command_mm:
            raise ValueError("gap outside configured command range")
        print(f'Gap={gap:g} mm is allowed; servo conversion occurs in MCU')
    else:
        print('Use --width OBJECT_MM to preview the transmitted net-opening command.')


if __name__ == '__main__':
    main()
