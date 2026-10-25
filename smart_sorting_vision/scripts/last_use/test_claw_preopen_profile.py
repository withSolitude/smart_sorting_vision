"""Offline tests. All dimensions below are SYNTHETIC, never install as calibration."""
from __future__ import annotations
import ast
import copy
import math
import yaml
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vision.grasp.claw_preopen_profile import ClawPreopenProfile, firmware_header
from communication.claw_preopen_transaction import latch_preopen
from communication.sorting_protocol import Mode, SortingFrame

SYNTHETIC = {"schema_version": 1, "calibrated": True, "margin_mm": 3.0,
             "samples": [{"servo_deg": 30.0, "gap_mm": 5.0},
                         {"servo_deg": 15.0, "gap_mm": 20.0},
                         {"servo_deg": 0.0, "gap_mm": 40.0}]}


class FakeLink:
    def __init__(self, response):
        self.sent = []
        self.response = response
    def send_frame(self, frame):
        self.sent.append(frame)
        return b"test"
    def poll(self):
        if self.sent and self.response:
            out, self.response = self.response, []
            return out
        return []
    def describe(self, frame):
        return frame.mode.name


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.p = ClawPreopenProfile(copy.deepcopy(SYNTHETIC))
    def test_missing_calibration(self):
        with self.assertRaises(ValueError):
            ClawPreopenProfile({"calibrated": False})
    def test_interpolation(self):
        self.assertEqual(self.p.angle_for_gap(5), 30)
        self.assertEqual(self.p.angle_for_gap(20), 15)
        self.assertAlmostEqual(self.p.angle_for_gap(30), 7.5)
        self.assertEqual(self.p.angle_for_gap(40), 0)
    def test_no_extrapolation(self):
        for gap in [0, 4.9, 40.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): self.p.angle_for_gap(gap)
    def test_width_margin_rounding(self):
        self.assertEqual(self.p.command_for_width(1), 5)
        self.assertEqual(self.p.command_for_width(20.01), 23.1)
        self.assertEqual(self.p.command_for_width(37), 40)
        for width in [37.01, 100, -1, 0, float('nan')]:
            with self.assertRaises(ValueError): self.p.command_for_width(width)
    def test_invalid_measurements(self):
        for key, val in [('gap_mm', float('nan')), ('servo_deg', 140),
                         ('gap_mm', 5), ('servo_deg', -1)]:
            data = copy.deepcopy(SYNTHETIC)
            data['samples'][1][key] = val
            with self.assertRaises(ValueError): ClawPreopenProfile(data)
    def test_limits_fit_protocol(self):
        data = copy.deepcopy(SYNTHETIC)
        data['samples'][0]['gap_mm'] = 5.03
        data['samples'][-1]['gap_mm'] = 39.98
        p = ClawPreopenProfile(data)
        self.assertEqual((p.min_command_mm, p.max_command_mm), (5.1, 39.9))
    def test_ack_then_pick_permitted(self):
        link = FakeLink([SortingFrame(mode=Mode.MCU_PICK_PREOPEN_ACK, x_mm=23.1)])
        self.assertEqual(latch_preopen(link, 23.01), 23.1)
        self.assertEqual([f.mode for f in link.sent], [Mode.PICK_PREOPEN])
    def test_bad_ack_busy_stop_error(self):
        for mode in [Mode.MCU_BUSY, Mode.MCU_ERROR, Mode.MCU_STOP_WORK, Mode.MCU_STOP_ACK,
                     Mode.MCU_PICK_PREOPEN_ACK]:
            link = FakeLink([SortingFrame(mode=mode, x_mm=7)])
            with self.assertRaises(RuntimeError): latch_preopen(link, 23)
            self.assertNotIn(Mode.PICK_TARGET, [f.mode for f in link.sent])
    def test_stop_after_ack_same_batch_wins(self):
        link = FakeLink([SortingFrame(mode=Mode.MCU_PICK_PREOPEN_ACK, x_mm=23),
                         SortingFrame(mode=Mode.MCU_STOP_WORK)])
        with self.assertRaises(RuntimeError): latch_preopen(link, 23)
    def test_timeout(self):
        with self.assertRaises(TimeoutError): latch_preopen(FakeLink([]), 23, .005)
    def test_real_single_pick_calls_latch_before_pick(self):
        # Execute the actual function with a stubbed failing latch. No CV/GPU/serial needed.
        path = ROOT / 'scripts/last_use/validation/validate_single_object_pick_to_tray.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_pick_first_stage')
        env = {'STM32Link': object, 'normalize_axial_deg': lambda x: x,
               'latch_preopen': lambda *a: (_ for _ in ()).throw(RuntimeError('no ACK'))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        link = FakeLink([])
        with self.assertRaises(RuntimeError):
            env['execute_pick_first_stage'](link, {'x_mm': -30,'y_mm': 5,'yaw_deg': 20,'preopen_mm': 23},
                                             ack_timeout_s=3, ready_timeout_s=30)
        self.assertEqual(link.sent, [])
    def test_actual_pipeline_method_and_reject_handler(self):
        path = ROOT / 'vision/pipeline/top_pick_pipeline.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TopPickPipeline')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_preopen_for_width')
        env = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        dummy = type('Pipeline', (), {'dynamic_preopen_enabled': True, 'claw_preopen_profile': self.p})()
        self.assertEqual(env['_preopen_for_width'](dummy, 20), 23)
        with self.assertRaises(ValueError): env['_preopen_for_width'](dummy, 38)
        with self.assertRaises(ValueError): env['_preopen_for_width'](dummy, None)
        source = path.read_text(encoding='utf-8')
        self.assertIn('reject("preopen_out_of_calibrated_range"', source)
    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler not available')
    def test_cpp_matches_python_and_uncalibrated_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            source = '#include "claw_preopen_profile.h"\n#include <cstdio>\nint main(){float a=999;'
            for g in [4.9, 5, 12.5, 20, 30, 40, 40.1]:
                source += 'if(ClawPreopen::ToAngle(%.9ff,a)) printf("ok %%.9f\\n",a);else puts("blocked");' % g
            source += 'return 0;}'
            (d/'test.cpp').write_text(source)
            for data in [SYNTHETIC, {'calibrated': False}]:
                (d/'claw_preopen_profile.h').write_text(firmware_header(data))
                subprocess.run(['g++','-std=c++11','-Wall','-Wextra','-Werror',str(d/'test.cpp'),'-o',str(d/'test')], check=True)
                lines = subprocess.check_output([str(d/'test')], text=True).splitlines()
                if data.get('calibrated'):
                    for g, line in zip([4.9,5,12.5,20,30,40,40.1], lines):
                        if not 5<=g<=40: self.assertEqual(line,'blocked')
                        else: self.assertAlmostEqual(float(line.split()[1]),self.p.angle_for_gap(g),places=5)
                else: self.assertEqual(lines, ['blocked']*7)


class TestInstalledCadProfile(unittest.TestCase):
    def setUp(self):
        runtime = yaml.safe_load((ROOT/'config/runtime_config.yaml').read_text(encoding='utf-8'))
        self.data = runtime['vision']['top']['grasp_orientation']['dynamic_preopen']
        self.p = ClawPreopenProfile.from_runtime_config(self.data)

    def test_yaml_is_the_only_pc_runtime_profile(self):
        self.assertEqual(self.p.kind, 'cad_four_bar')
        self.assertFalse(self.data['hardware_validated'])
        self.assertEqual(self.data['closed_servo_deg'], -15)
        self.assertEqual(self.data['standby_servo_deg'], -95)
        self.assertEqual(self.data['margin_mm'], 5)
        self.assertFalse((ROOT/'config/claw_preopen_profile.json').exists())
        with self.assertRaises(ValueError): self.p.angle_for_gap(20)

    def test_model_boundaries_and_large_objects(self):
        self.assertEqual((self.p.min_command_mm, self.p.max_command_mm), (.9, 67.5))
        self.assertEqual(self.p.command_for_width(23), 28)
        self.assertEqual(self.p.command_for_width(62.5), 67.5)
        with self.assertRaises(ValueError): self.p.command_for_width(62.51)

    def test_invalid_yaml_profile_blocked(self):
        for key, value in [('closed_servo_deg', -140), ('standby_servo_deg', -150),
                           ('configured', False), ('margin_mm', 0),
                           ('min_command_mm', 70), ('profile_kind', 'legacy')]:
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.assertRaises(ValueError): ClawPreopenProfile.from_runtime_config(data)

    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler not available')
    def test_mcu_header_accepts_exact_yaml_range(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            shutil.copy2(ROOT.parent/'Control_fixed/User_File/3_Robot/claw_preopen_profile.h',
                         d/'claw_preopen_profile.h')
            (d/'test.cpp').write_text('''#include "claw_preopen_profile.h"
#include <cstdio>
#include <limits>
int main() {
  for(int tick=9; tick<=675; ++tick) {
    float a=999;
    if(!ClawPreopen::ToAngle(tick/10.0f,a)) return 1;
    if(a < -95 || a >= -15) return 2;
    std::printf("%.9f\\n",a);
  }
  float a=999;
  if(ClawPreopen::ToAngle(67.6f,a) || a!=999) return 3;
  if(ClawPreopen::ToAngle(std::numeric_limits<float>::quiet_NaN(),a)) return 4;
  return 0;
}''', encoding='utf-8')
            subprocess.run(['g++','-std=c++11','-Wall','-Wextra','-Werror',str(d/'test.cpp'),'-o',str(d/'test')],check=True)
            lines = subprocess.check_output([str(d/'test')],text=True).splitlines()
            self.assertEqual(len(lines),667)
            values = [float(line) for line in lines]
            self.assertTrue(all(-95 <= value < -15 for value in values))
            self.assertTrue(all(a > b for a,b in zip(values,values[1:])))

if __name__ == '__main__':
    unittest.main()
