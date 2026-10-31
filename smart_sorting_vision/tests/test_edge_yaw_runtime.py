"""最终下发复核：下降补偿跨入边缘区时跳过候选，随后仍能继续抓取。"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from communication.sorting_protocol import Mode
from runtime.sorting_runtime import SortingRuntime, RuntimeState
from vision.calibration.yaw_servo import YawServoCalibration
from vision.grasp.edge_yaw_limit import EdgeYawLimit


class EdgeYawRuntimeTests(unittest.TestCase):
    def test_drift_into_edge_is_rejected_before_preopen_and_next_candidate_can_run(self):
        runtime = SortingRuntime.__new__(SortingRuntime)
        runtime.running = True
        runtime.state = RuntimeState.WAIT_TOP_OBJECT
        runtime.completed_count = 0
        runtime.minimum_items_before_empty_finish = 12
        runtime.pick_x_limits_mm = runtime.pick_y_limits_mm = (-300, 0)
        runtime.yaw_servo_calibration = YawServoCalibration(-63.280987, 0.232430, 1.04720792, -90, 90)
        result = SimpleNamespace(ready=True, object_count=1, x_mm=-22, y_mm=-100,
                                 yaw_deg=60, preopen_mm=20, grasp_width_mm=12)
        runtime.top_camera = SimpleNamespace(read=lambda: np.zeros((4, 4, 3), np.uint8))
        runtime.top_pipeline = SimpleNamespace(
            process=lambda image: result, reset=Mock(), edge_yaw_limit=EdgeYawLimit(enabled=True),
        )
        runtime.descent_drift_compensator = SimpleNamespace(apply=lambda x, y, yaw: SimpleNamespace(
            x_mm=-10, y_mm=y, predicted_drift_x_mm=-12, predicted_drift_y_mm=0,
            command_delta_x_mm=12, command_delta_y_mm=0,
        ))
        runtime._action_still_allowed = Mock(return_value=True)
        runtime._send_command = Mock(return_value=True)
        runtime._enter_state = Mock()
        runtime._fail = Mock()
        runtime.process_top()
        runtime._send_command.assert_not_called()
        runtime._fail.assert_not_called()
        runtime.top_pipeline.reset.assert_called_once()
        self.assertEqual(runtime.state, RuntimeState.WAIT_TOP_OBJECT)
        self.assertIsNone(runtime.current_pick_command)

        # 下一帧得到合法姿态即可继续，不需要重启比赛或重新START。
        runtime.descent_drift_compensator = None
        result.x_mm, result.yaw_deg = -10, 0
        runtime.process_top()
        runtime._fail.assert_not_called()
        self.assertEqual(runtime._send_command.call_args.args[0].mode, Mode.PICK_PREOPEN)
        runtime._enter_state.assert_called_once_with(RuntimeState.WAIT_PICK_PREOPEN_ACK)


if __name__ == "__main__":
    unittest.main()
