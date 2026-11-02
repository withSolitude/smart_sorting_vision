"""按比例缩放 pixel_to_robot.json 的机械坐标输出。

什么时候用
----------
修正了 `xy_motion_config.h` 的 `Travel_Per_Rev_A/B_mm` 之后，旧标定文件记录的
是**修正前控制器读数**：jog 到物理点 P 时控制器以为自己在 P/k（k = 真实/旧值），
所以文件里的坐标整体小了 k 倍。

若这次误差是纯均匀尺度（本次实测：命令 150mm 实走 160mm，k = 160/150），
把单应的线性部分和 tcp_offset 同乘 k，等价于按新尺度重标，不必重跑相机。

    H_new = diag(k, k, 1) @ H_old
    tcp_new = k * tcp_old

这不修正旋转或平移误差，只修正尺度。若重投影误差本来就大，请改用
`python -m scripts.last_debug.calibrate_pixel_to_robot --verify-point` 重标。

用法
----
    # 只看会发生什么
    python -m scripts.last_debug.scale_pixel_to_robot --factor 1.0667 --dry-run

    # 应用（先自动备份）
    python -m scripts.last_debug.scale_pixel_to_robot --factor 1.0667 --apply

    # 撤销
    python -m scripts.last_debug.scale_pixel_to_robot --restore

说明
----
脚本用 metadata.scale_fix 做幂等保护，同一个系数不会重复应用；
180° 旋转那种自逆变换不适用这里的逻辑（那是 rotate_180，不是缩放）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime.app_config import load_runtime_config

    return load_runtime_config(ROOT, config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="按比例缩放 pixel_to_robot.json")
    parser.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    parser.add_argument("--factor", type=float, default=1.0667,
                        help="真实距离 / 旧标定距离，例如 160/150 = 1.0667")
    parser.add_argument("--apply", action="store_true", help="写回文件（先备份）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写")
    parser.add_argument("--restore", action="store_true", help="用最近备份还原")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    path = Path(cfg["calibration"]["pixel_to_robot"])
    if not path.exists():
        raise SystemExit(f"找不到标定文件: {path}")

    if args.restore:
        backups = sorted(path.parent.glob(f"{path.name}.bak-*"))
        if not backups:
            raise SystemExit(f"没有备份: {path.parent}")
        latest = backups[-1]
        if not args.yes and input(f"用 {latest.name} 还原？(yes): ").strip().lower() != "yes":
            return
        shutil.copy2(latest, path)
        print(f"[RESTORE] {path} <- {latest.name}")
        return

    k = float(args.factor)
    if not np.isfinite(k) or k <= 0:
        raise SystemExit("--factor 必须是正的有限数值")

    data = json.loads(path.read_text(encoding="utf-8"))
    H = np.asarray(data["homography"], dtype=np.float64)
    if H.shape != (3, 3):
        raise SystemExit(f"homography 形状异常: {H.shape}")

    previous = (data.get("metadata") or {}).get("scale_fix")
    if previous and abs(float(previous.get("factor", 0.0)) - k) < 1e-9 and args.apply:
        raise SystemExit(
            "该文件已经按同一系数缩放过，再次应用会重复放大。"
            "确认要再乘一次请手动改 JSON 或换 --factor。"
        )

    S = np.diag([k, k, 1.0])
    H_new = S @ H
    tcp_old = (float(data.get("tcp_offset_x_mm", 0.0)), float(data.get("tcp_offset_y_mm", 0.0)))
    tcp_new = (tcp_old[0] * k, tcp_old[1] * k)

    print("=" * 70)
    print(f"文件    : {path}")
    print(f"系数 k  : {k:.6f}")
    print(f"H 线性部分: {np.round(H[:2, :2], 6).tolist()}")
    print(f"        -> {np.round(H_new[:2, :2], 6).tolist()}")
    print(f"tcp_offset: {tcp_old} -> {tcp_new}")

    # 抽样自检：把 4 个角点走一遍，展示缩放后的机械坐标
    sample = [(600.0, 200.0), (1600.0, 200.0), (1600.0, 900.0), (600.0, 900.0)]
    for u, v in sample:
        p_old = H @ np.array([u, v, 1.0])
        p_new = H_new @ np.array([u, v, 1.0])
        print(f"  pixel({u:.0f},{v:.0f}) 旧=({p_old[0]/p_old[2]:8.2f},{p_old[1]/p_old[2]:8.2f})"
              f"  新=({p_new[0]/p_new[2]:8.2f},{p_new[1]/p_new[2]:8.2f})")
    print("=" * 70)

    if not args.apply:
        print("[DRY-RUN] 未写入。确认无误后加 --apply。")
        return
    if not args.yes and input("确认写入？(yes): ").strip().lower() != "yes":
        return

    backup = path.parent / f"{path.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    metadata = dict(data.get("metadata") or {})
    metadata["scale_fix"] = {
        "factor": k,
        "applied_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "Travel_Per_Rev corrected; old calibration was in pre-fix units",
    }
    data["homography"] = H_new.tolist()
    data["tcp_offset_x_mm"] = tcp_new[0]
    data["tcp_offset_y_mm"] = tcp_new[1]
    data["metadata"] = metadata
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[BACKUP] {backup.name}")
    print(f"[SAVE]   {path}")
    print("下一步：跑一次单件抓取确认落点，如果反而变短说明方向反了（k 应取 150/160）。")


if __name__ == "__main__":
    main()
