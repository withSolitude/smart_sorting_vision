"""现场旧配置可直接启用名义选轴，开关类型错误在启动前暴露。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from runtime.app_config import ConfigError, load_runtime_config


ROOT = Path(__file__).resolve().parents[1]


class GraspAxisConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = yaml.safe_load(
            (ROOT / "config" / "runtime_config.yaml").read_text(encoding="utf-8")
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "runtime.yaml"

    def load(self, raw):
        self.path.write_text(
            yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
        )
        return load_runtime_config(ROOT, self.path)

    def collision(self, raw):
        return raw["vision"]["top"]["gripper_collision"]

    def test_legacy_field_config_defaults_on_without_changing_tuned_values(self):
        collision = self.collision(self.raw)
        collision.pop("axis_selection", None)
        collision["enabled"] = False
        # 模拟用户已调好的旧现场配置，不用仓库默认值覆盖它。
        collision["grasp_search"]["enabled"] = True
        collision["grasp_search"]["yaw_offsets_deg"] = [0, 90, -15]
        collision["grasp_search"]["longitudinal_offsets_mm"] = [0.0]
        collision["grasp_search"]["origin_penalty_per_mm"] = 0.09
        before = deepcopy(collision)
        loaded = self.collision(self.load(self.raw))
        self.assertEqual(loaded.pop("axis_selection"), {"enabled": True})
        self.assertEqual(loaded, before)

    def test_empty_optional_section_defaults_on(self):
        self.collision(self.raw)["axis_selection"] = {}
        loaded = self.collision(self.load(self.raw))
        self.assertIs(loaded["axis_selection"]["enabled"], True)

    def test_explicit_booleans_are_independent_of_full_collision_switch(self):
        for full_collision in (False, True):
            for axis_selection in (False, True):
                with self.subTest(full=full_collision, axis=axis_selection):
                    raw = deepcopy(self.raw)
                    self.collision(raw)["enabled"] = full_collision
                    self.collision(raw)["axis_selection"] = {
                        "enabled": axis_selection
                    }
                    loaded = self.collision(self.load(raw))
                    self.assertIs(loaded["enabled"], full_collision)
                    self.assertIs(
                        loaded["axis_selection"]["enabled"], axis_selection
                    )

    def test_non_mapping_section_is_rejected(self):
        for value in (None, False, "false", [], 1):
            with self.subTest(value=value):
                raw = deepcopy(self.raw)
                self.collision(raw)["axis_selection"] = value
                with self.assertRaisesRegex(
                    ConfigError, r"gripper_collision\.axis_selection 必须是对象"
                ):
                    self.load(raw)

    def test_enabled_accepts_only_booleans(self):
        for value in (None, "false", "true", 0, 1, [], {}):
            with self.subTest(value=value):
                raw = deepcopy(self.raw)
                self.collision(raw)["axis_selection"] = {"enabled": value}
                with self.assertRaisesRegex(
                    ConfigError,
                    r"gripper_collision\.axis_selection\.enabled 必须是 true/false",
                ):
                    self.load(raw)


if __name__ == "__main__":
    unittest.main()
