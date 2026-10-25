"""Install the CAD claw patch, preserving unrelated XY/Yaw/TCP/config changes.

Supports the uploaded original code and the previous measured-profile patch.
Checks all files before writing; backs up changed files; never flashes firmware.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil


def edit_chunks(source, variants):
    for chunks in variants:
        candidate = source
        for chunk in chunks:
            old, new = chunk['old'], chunk['new']
            if candidate.count(new) == 1:
                continue
            if candidate.count(old) == 1:
                candidate = candidate.replace(old, new, 1)
            else:
                break
        else:
            return candidate
    raise ValueError('Required code blocks differ; send this current file for a merge')


def edit_claw_header(source):
    include = '#include "claw_preopen_profile.h"'
    if include not in source:
        anchor = '#include "2_Device/TOF050F/dvc_tof050f.h"'
        if source.count(anchor) != 1:
            raise ValueError('Could not locate holding_jaw.h include anchor')
        source = source.replace(anchor, anchor+'\n'+include, 1)
    names = ['Grap_Target_Pre_Clip_Angle', 'Recognition_Claw_Angle',
             'Claw_Servo_Motor_Angle_Disable', 'Claw_Servo_Motor_Angle_Pre_Clip',
             'Claw_Servo_Motor_Angle_Clip', 'Claw_Servo_Motor_Angle_Release',
             'Claw_Servo_Motor_Target_Angle']
    for name in names:
        pattern = r'(^[ \t]*float[ \t]+'+re.escape(name)+r'[ \t]*=\s*)[^;]+;'
        symbol = 'kClosedAngleDeg' if name == 'Claw_Servo_Motor_Angle_Clip' else 'kStandbyAngleDeg'
        source, count = re.subn(pattern, lambda m: m[1]+'ClawPreopen::'+symbol+' * PI / 180.0f;', source, flags=re.M)
        if count != 1:
            raise ValueError(f'Expected one active initializer for {name}, found {count}')
    old = '// 本周期真正使用的预收缩舵机角。无动态宽度时等于旧固定值。'
    source = source.replace(old, '// 本周期预开口角由 CAD 净开口表提供；初始保持已确认的待机张开角。')
    old = '// 动态预收缩：基于夹爪CAD几何的“有效净开口(mm)” -> 舵机角度映射'
    source = source.replace(old, '// 新夹爪动态预开口：claw_preopen_profile.h 中的 CAD 四连杆净开口表。\n'
        '    // -15° 近闭合、-95° 待机张开；角度越负越张开。控制接口仍使用 rad。\n'
        '    // 下面旧 CAD 参数仅留作历史记录，不参与任何预开口计算。\n'
        '    // 历史说明：基于旧夹爪CAD几何的“有效净开口(mm)” -> 舵机角度映射')
    return source if source.endswith('\n') else source+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side', choices=['upper', 'lower'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--apply', action='store_true', help='Write after preflight and backup; otherwise just inspect')
    args = p.parse_args()
    root, package = args.root.expanduser().resolve(), Path(__file__).resolve().parent
    marker = 'main.py' if args.side == 'upper' else 'User_File/3_Robot/holding_jaw.cpp'
    if not (root/marker).is_file():
        p.error(f'Wrong project root: missing {marker}')
    manifest = json.loads((package/'changes.json').read_text(encoding='utf-8'))[args.side]
    pending = []
    for entry in manifest:
        rel = Path(entry['path'])
        target = root/rel
        if rel.is_absolute() or '..' in rel.parts or not target.resolve().is_relative_to(root):
            p.error(f'Unsafe target: {rel}')
        old = target.read_bytes() if target.exists() else None
        try:
            if entry['kind'] == 'file':
                wanted = (package/'files'/args.side/rel).read_bytes()
                if old is not None and old != wanted:
                    if hashlib.sha256(old).hexdigest() not in entry.get('prior_sha256', []):
                        raise ValueError('Existing custom file differs; not replacing it automatically')
            else:
                if old is None:
                    raise ValueError('Required source file is missing')
                source = old.decode('utf-8')
                newline = '\r\n' if '\r\n' in source else '\n'
                source = source.replace('\r\n', '\n')
                if entry['kind'] == 'claw_header':
                    source = edit_claw_header(source)
                else:
                    source = edit_chunks(source, entry['variants'])
                wanted = source.replace('\n', newline).encode('utf-8')
        except (ValueError, UnicodeError) as exc:
            p.error(f'{rel}: {exc}. No files have been changed.')
        if old != wanted:
            pending.append((rel, target, old, wanted))
            print(f'[CHANGE] {rel}')
    print(f'[CHECK OK] {len(pending)} file(s). CAD close=-15 deg, standby/release=-95 deg.')
    print('Unrelated yaw/XY/TCP/runtime_config settings are preserved.')
    if not args.apply or not pending:
        print('Add --apply to install.' if pending else 'Already installed.')
        return 0
    backup = root/'preopen_patch_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True, exist_ok=False)
    record = {'side': args.side, 'files': []}
    for rel, target, old, wanted in pending:
        if old is not None:
            dest = backup/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, dest)
        record['files'].append({'path': rel.as_posix(), 'created_new': old is None})
    (backup/'restore_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    for rel, target, old, wanted in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(wanted)
        print(f'[UPDATED] {rel}')
    print(f'Backup: {backup}')
    print('Install both sides, then rebuild/flash MCU. First verify empty claw opening at a safe height.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
