from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from typing import Any


class SideRecognitionConsensus:
    """
    侧面识别多帧一致性确认器。

    输入：
        SideRecognizer.recognize() 的单帧结果。

    默认策略：
        最近 5 帧中，至少 4 帧的完整结果完全一致：

            (color, qr_present, shape)

        如果启用二维码内容校验，则比较键追加：

            qr_content

        才认为本次识别稳定。

    注意：
        这里是“完整组合一致”，不是分别对颜色、二维码、形状投票。

        例如下面这种情况不会被错误拼接：

            frame1: red,  QR=yes, cube
            frame2: red,  QR=no,  cube
            frame3: blue, QR=yes, cube
            ...

        必须同一个完整三元组出现足够次数。

    如果你后面想严格要求 5 帧全部一样，只需要：

        required_hits=5
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        required_hits: int = 4,
        check_qr_content: bool = False,
    ) -> None:

        window_size = int(window_size)
        required_hits = int(required_hits)

        if window_size <= 0:
            raise ValueError(
                "window_size 必须 > 0"
            )

        if required_hits <= 0:
            raise ValueError(
                "required_hits 必须 > 0"
            )

        if required_hits > window_size:
            raise ValueError(
                "required_hits 不能大于 window_size"
            )

        self.window_size = window_size
        self.required_hits = required_hits
        self.check_qr_content = bool(check_qr_content)

        self.history = deque(
            maxlen=window_size
        )

        # 一旦确认成功，锁存当前物体结果。
        # 直到上层在“物体被拿走”时调用 reset()。
        self.latched_result = None

    def reset(
        self,
    ) -> None:

        self.history.clear()
        self.latched_result = None

    def _candidate_key(
        self,
        result: dict[str, Any] | None,
    ):
        """
        只有颜色和形状都有效时才形成候选。

        qr_present=False 是合法结果，
        所以不能把 False 当作无效。
        """

        if result is None:
            return None

        if not result.get(
            "object_found",
            False,
        ):
            return None

        color = str(
            result.get(
                "color",
                "unknown",
            )
        )

        shape = str(
            result.get(
                "shape",
                "unknown",
            )
        )

        if (
            color == "unknown"
            or
            shape == "unknown"
        ):
            return None

        qr_present = bool(
            result.get(
                "qr_present",
                False,
            )
        )

        key = [
            color,
            qr_present,
            shape,
        ]

        if self.check_qr_content:
            # QRDetector 成功时应当有非空文本。若内容缺失，不能把
            # “无法确认内容”当成合法内容参与 READY。
            qr_content = str(result.get("qr_content", "") or "")
            if qr_present and not qr_content:
                return None
            key.append(qr_content if qr_present else "")

        return tuple(key)

    def update(
        self,
        result: dict[str, Any] | None,
    ) -> dict[str, Any]:

        # 已确认后直接保持结果，避免同一个物体反复变化。
        if self.latched_result is not None:

            return {
                "ready":
                    True,

                "latched":
                    True,

                "votes":
                    self.required_hits,

                "window":
                    len(
                        self.history
                    ),

                "required_hits":
                    self.required_hits,

                "candidate": self._candidate_key(self.latched_result),

                "result":
                    deepcopy(
                        self.latched_result
                    ),
            }

        key = self._candidate_key(
            result
        )

        self.history.append(
            {
                "key":
                    key,

                "result":
                    deepcopy(
                        result
                    )
                    if result is not None
                    else None,
            }
        )

        valid_keys = [
            item["key"]
            for item in self.history
            if item["key"] is not None
        ]

        if not valid_keys:

            return {
                "ready":
                    False,

                "latched":
                    False,

                "votes":
                    0,

                "window":
                    len(
                        self.history
                    ),

                "required_hits":
                    self.required_hits,

                "candidate":
                    None,

                "result":
                    None,
            }

        counts = Counter(
            valid_keys
        )

        best_key, best_votes = (
            counts.most_common(
                1
            )[0]
        )

        ready = (
            best_votes
            >=
            self.required_hits
        )

        final_result = None

        if ready:

            matching_results = [
                item["result"]
                for item in self.history
                if (
                    item["key"]
                    ==
                    best_key
                    and
                    item["result"]
                    is not None
                )
            ]

            # 用最新一帧同类结果作为 details 基础。
            final_result = deepcopy(
                matching_results[-1]
            )

            color_confidences = [
                float(
                    item.get(
                        "color_confidence",
                        0.0,
                    )
                )
                for item in matching_results
            ]

            shape_confidences = [
                float(
                    item.get(
                        "shape_confidence",
                        0.0,
                    )
                )
                for item in matching_results
            ]

            final_result[
                "color_confidence"
            ] = (
                sum(
                    color_confidences
                )
                /
                max(
                    len(
                        color_confidences
                    ),
                    1,
                )
            )

            final_result[
                "shape_confidence"
            ] = (
                sum(
                    shape_confidences
                )
                /
                max(
                    len(
                        shape_confidences
                    ),
                    1,
                )
            )

            final_result[
                "consensus"
            ] = {
                "window_size":
                    self.window_size,

                "required_hits":
                    self.required_hits,

                "votes":
                    int(
                        best_votes
                    ),
            }

            self.latched_result = deepcopy(
                final_result
            )

        return {
            "ready":
                bool(
                    ready
                ),

            "latched":
                False,

            "votes":
                int(
                    best_votes
                ),

            "window":
                len(
                    self.history
                ),

            "required_hits":
                self.required_hits,

            "candidate":
                best_key,

            "result":
                final_result,
        }
