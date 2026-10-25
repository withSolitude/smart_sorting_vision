"""Offline CAD -> paired claw lookup files; never opens a camera or serial port.

Runtime does NOT need CadQuery. Only regeneration from STEP needs CadQuery/numpy.
This geometry belongs to the user's jaizhau.stp, not an arbitrary parallel claw.
"""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vision.grasp.claw_preopen_profile import ClawPreopenProfile, firmware_header

# STEP hole axis centers in assembly (Y,Z), mm. O: gear, A: follower ground,
# B: crank-finger pin, C: follower-finger pin. Rotation axes are assembly X.
GEOMETRY = [
    ((10.40115851389, 10.15556399232), (18.5, -2.5),
     (7.111714077, 39.974677918), (15.210488174, 27.319106491)),
    ((-16.1, 10.0), (-24.2, -2.5),
     (-9.835759961, 39.338699643), (-18.278164456, 26.909723967)),
]


class CadClaw:
    def __init__(self, step: Path):
        import cadquery as cq
        import numpy as np
        self.np = np
        self.parts = cq.importers.importStep(str(step)).solids().vals()
        if len(self.parts) != 9:
            raise ValueError("Expected the supplied 9-solid jaizhau.stp assembly")
        # Guard the solid order/geometry, including the inner curved finger faces.
        expected = [(18.7, 6.81293536605, 52.5572410912),
                    (18.7, -9.19242380495, 51.9083568906)]
        for solid, center in zip(self.parts[7:9], expected):
            if np.linalg.norm(np.array(solid.Center().toTuple())-center) > .001:
                raise ValueError("STEP geometry/order differs; this linkage must be re-extracted")

    def pose(self, side: int, opening_deg: float):
        np = self.np
        o, a, b0, c0 = (np.array(p) for p in GEOMETRY[side])
        angle = math.radians(opening_deg * (-1 if side == 0 else 1))
        co, si = math.cos(angle), math.sin(angle)
        b = o + np.array([[co, -si], [si, co]]) @ (b0-o)
        v = b-a
        d = float(np.linalg.norm(v))
        follower, coupler = float(np.linalg.norm(c0-a)), float(np.linalg.norm(c0-b0))
        along = (follower*follower-coupler*coupler+d*d)/(2*d)
        h2 = follower*follower-along*along
        if h2 < -1e-8:
            raise ValueError("Four-bar branch is unreachable")
        e = v/d
        n = np.array([-e[1], e[0]])
        cross = lambda p, q: p[0]*q[1]-p[1]*q[0]
        # Preserve the assembly branch instead of choosing the nearer random root.
        branch = 1 if cross(b0-a, c0-a) > 0 else -1
        c = a + along*e + branch*math.sqrt(max(0., h2))*n
        v0, v1 = c0-b0, c-b
        twist = math.atan2(cross(v0, v1), float(np.dot(v0, v1)))
        return b, c, twist

    def finger(self, side: int, opening_deg: float):
        b0 = self.np.array(GEOMETRY[side][2])
        b, _, twist = self.pose(side, opening_deg)
        return self.parts[7+side].rotate((0., *b0), (1., *b0), math.degrees(twist)).translate((0., *(b-b0)))

    def gap(self, opening_deg: float) -> float:
        # A conservative straight corridor across the entire fingers, including
        # curved edges: this is NOT the larger gap at the concave face center.
        right = self.finger(0, opening_deg).BoundingBox()
        left = self.finger(1, opening_deg).BoundingBox()
        return right.ymin-left.ymax

    def closed_opening_deg(self) -> float:
        lo, hi = -5., 5.
        if not self.gap(lo) < 0 < self.gap(hi):
            raise ValueError("Supplied assembly does not bracket the expected closure")
        for _ in range(45):
            mid = (lo+hi)/2
            if self.gap(mid) > 0:
                hi = mid
            else:
                lo = mid
        return (lo+hi)/2


def build(step: Path, closed: float = -15., standby: float = -95., margin: float = 3.) -> dict:
    if not all(math.isfinite(v) for v in (closed, standby, margin)):
        raise ValueError("Finite parameters required")
    if not -135 <= standby < closed <= 135 or not 5 <= closed-standby <= 80:
        raise ValueError("This CAD builder supports a 5..80 degree opening travel within servo limits")
    claw = CadClaw(step)
    zero = claw.closed_opening_deg()
    # 0.5 deg table. 0.2 mm deducted from geometric clearances covers lookup
    # interpolation and the existing 270-deg servo driver's PWM rounding.
    # It is not a bound on assembly/servo accuracy; margin remains separate.
    reserve = .2
    n = math.ceil(((closed-standby)-1.)/.5)
    angles = [closed-1.-((closed-standby)-1.)*i/n for i in range(n+1)]
    samples = [{"servo_deg": round(q, 9),
                "gap_mm": round(claw.gap(zero+closed-q)-reserve, 9)} for q in angles]
    data = {
        "schema_version": 2, "profile_kind": "cad_four_bar", "configured": True,
        "hardware_validated": False,
        "source": "jaizhau.stp four-bar kinematics; user closed=-15 deg, standby=-95 deg",
        "source_step_sha256": hashlib.sha256(step.read_bytes()).hexdigest(),
        "closed_servo_deg": closed, "standby_servo_deg": standby,
        "closed_gap_assumed_mm": 0., "cad_closed_opening_deg": zero,
        "gap_definition": "min(right finger Y) - max(left finger Y), full solid envelope",
        "gear_ratio": 1., "crank_length_mm": 30., "follower_length_mm": 30.,
        "finger_pin_distance_mm": 15.0251,
        "lookup_reserve_mm": reserve, "margin_mm": margin,
        "samples": samples,
    }
    ClawPreopenProfile(data)
    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=Path, required=True)
    p.add_argument("--closed-deg", type=float, default=-15.)
    p.add_argument("--standby-deg", type=float, default=-95.)
    p.add_argument("--margin-mm", type=float, default=3.)
    p.add_argument("--out-dir", type=Path, required=True, help="New output directory; existing files never overwritten")
    args = p.parse_args()
    data = build(args.step, args.closed_deg, args.standby_deg, args.margin_mm)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    runtime = {key: data[key] for key in (
        "profile_kind", "configured", "hardware_validated", "closed_servo_deg",
        "standby_servo_deg", "margin_mm")}
    profile = ClawPreopenProfile(data)
    runtime["enabled"] = True
    runtime["min_command_mm"] = profile.min_command_mm
    runtime["max_command_mm"] = profile.max_command_mm
    (args.out_dir/'dynamic_preopen.yaml').write_text(
        yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding='utf-8')
    (args.out_dir/'claw_preopen_profile.h').write_text(firmware_header(data), encoding='utf-8')
    print(f"CAD lookup generated: {profile.min_command_mm:.1f}..{profile.max_command_mm:.1f} mm")
    print(f"Closed={args.closed_deg:g} deg; standby={args.standby_deg:g} deg; total width margin={args.margin_mm:g} mm")
    print(f"Outputs: {args.out_dir.resolve()}; hardware movement has not been tested")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
