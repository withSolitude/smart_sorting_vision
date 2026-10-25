from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from communication.sorting_protocol import (
    SortingProtocol,
    make_heartbeat,
    make_pick_target,
    make_sort_result,
)
from communication.stm32_link import (
    STM32Link,
)


def main() -> None:
    """
    不需要真实串口。

    专门测试：
        - 半包
        - 粘包
        - 前导垃圾字节
        - 错位后重新同步
    """

    link = STM32Link()

    raw_1 = SortingProtocol.pack(
        make_heartbeat()
    )

    raw_2 = SortingProtocol.pack(
        make_pick_target(
            -56.2,
            25.1,
        )
    )

    raw_3 = SortingProtocol.pack(
        make_sort_result(
            4
        )
    )

    # ------------------------------------------------------------
    # 1. 半包
    # ------------------------------------------------------------
    link.rx_buffer.extend(
        raw_1[:4]
    )

    assert (
        link._extract_frames()
        ==
        []
    )

    link.rx_buffer.extend(
        raw_1[4:]
    )

    frames = (
        link._extract_frames()
    )

    assert len(
        frames
    ) == 1

    # ------------------------------------------------------------
    # 2. 粘包
    # ------------------------------------------------------------
    link.rx_buffer.extend(
        raw_2
        +
        raw_3
    )

    frames = (
        link._extract_frames()
    )

    assert len(
        frames
    ) == 2

    # ------------------------------------------------------------
    # 3. 垃圾前缀
    # ------------------------------------------------------------
    link.rx_buffer.extend(
        b"\x00\x11\x22\x33"
        +
        raw_1
    )

    frames = (
        link._extract_frames()
    )

    assert len(
        frames
    ) == 1

    # ------------------------------------------------------------
    # 4. 错位/坏帧后恢复
    # ------------------------------------------------------------
    bad = bytearray(
        raw_1
    )

    bad[9] = 0x00

    link.rx_buffer.extend(
        bytes(
            bad
        )
        +
        raw_2
    )

    frames = (
        link._extract_frames()
    )

    assert len(
        frames
    ) == 1

    print(
        "=" * 64
    )

    print(
        "STM32 LINK PARSER TEST PASSED"
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()
