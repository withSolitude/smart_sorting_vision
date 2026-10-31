"""边缘限角在最终下发前仍生效；不打开串口、相机或模型。"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from runtime.sorting_runtime import RuntimeState, SortingRuntime
from scripts.last_use.test_top_pick_only_debug import make_frozen_target
from scripts.last_use.validation.validate_single_object_pick_to_tray import execute_pick_first_stage
from vision.calibration.yaw_servo import YawServoCalibration
from vision.grasp.edge_yaw_limit import EdgeYawLimit


class EdgeYawDispatchTests(unittest.TestCase):
    yaw = YawServoCalibration(-63.280987, 0.232430, 1.04720792, -90, 90)
    rule = EdgeYawLimit(True, -20, 40, 80, 60, 5)

    def runtime(self, x, axis, drift=None):
        result = SimpleNamespace(ready=True, object_count=1, x_mm=x, y_mm=-100,
                                 yaw_deg=axis, preopen_mm=30, grasp_width_mm=22)
        pipeline = SimpleNamespace(process=Mock(return_value=result), reset=Mock(), edge_yaw_limit=self.rule)
        runtime = SortingRuntime(
            SimpleNamespace(read=lambda: np.zeros((10, 10, 3), np.uint8)),
            None, pipeline, None, None, None, yaw_servo_calibration=self.yaw,
            descent_drift_compensator=drift,
            pick_x_limits_mm=(-300, 0), pick_y_limits_mm=(-300, 0),
        )
        runtime.running = True
        runtime.state = RuntimeState.WAIT_TOP_OBJECT
        runtime._send_command = Mock(return_value=True)
        runtime._fail = Mock()
        return runtime

    def test_runtime_rejects_unsafe_servo_before_preopen_without_error_state(self):
        runtime = self.runtime(-10, 20)  # 协议20°对应舵机约87.4°，不是安全舵机60°。
        runtime.process_top()
        runtime._send_command.assert_not_called()
        runtime._fail.assert_not_called()
        runtime.top_pipeline.reset.assert_called_once()
        self.assertEqual(runtime.state, RuntimeState.WAIT_TOP_OBJECT)
        self.assertIsNone(runtime.current_pick_command)

    def test_runtime_allows_actual_sixty_degree_servo_then_preopen(self):
        runtime = self.runtime(-10, -6)  # 量化后舵机约60.22°。
        runtime.process_top()
        runtime._send_command.assert_called_once()
        self.assertEqual(runtime._send_command.call_args.kwargs["label"], "PICK_PREOPEN")
        runtime._fail.assert_not_called()
        self.assertEqual(runtime.state, RuntimeState.WAIT_PICK_PREOPEN_ACK)

    def test_descent_compensation_cannot_push_an_unsafe_yaw_into_edge_region(self):
        drift = SimpleNamespace(apply=lambda x, y, angle: SimpleNamespace(
            x_mm=x + 2, y_mm=y, predicted_drift_x_mm=-2, predicted_drift_y_mm=0,
            command_delta_x_mm=2, command_delta_y_mm=0,
        ))
        runtime = self.runtime(-21, 20, drift)
        runtime.process_top()
        runtime._send_command.assert_not_called()
        runtime._fail.assert_not_called()
        self.assertEqual(runtime.state, RuntimeState.WAIT_TOP_OBJECT)

    def test_final_protocol_x_at_exact_boundary_is_not_the_edge_region(self):
        runtime = self.runtime(-19.96, 20)
        runtime.process_top()
        self.assertEqual(runtime.current_pick_command.x_mm, -20.0)
        runtime._send_command.assert_called_once()
        runtime._fail.assert_not_called()

    def test_space_grasp_helper_checks_post_drift_wire_point(self):
        ready = SimpleNamespace(ready=True, x_mm=-21, y_mm=-100, yaw_deg=20, preopen_mm=30,
                                grasp_pixel=None, candidates=[SimpleNamespace(score=1)])
        drift = SimpleNamespace(apply=lambda x, y, angle: SimpleNamespace(
            x_mm=x + 2, y_mm=y, command_delta_x_mm=2, command_delta_y_mm=0,
            predicted_drift_x_mm=-2, predicted_drift_y_mm=0,
        ))
        with self.assertRaisesRegex(ValueError, "边缘Yaw限制"):
            make_frozen_target(ready, self.yaw, drift, edge_yaw_limit=self.rule)

    def test_validation_rejects_before_any_serial_command(self):
        link = Mock()
        with self.assertRaisesRegex(ValueError, "边缘Yaw限制"):
            execute_pick_first_stage(
                link, {"x_mm": -10, "y_mm": -100, "yaw_deg": 20, "preopen_mm": 30},
                ack_timeout_s=1, ready_timeout_s=1,
                edge_yaw_limit=self.rule, yaw_calibration=self.yaw,
            )
        self.assertEqual(link.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
