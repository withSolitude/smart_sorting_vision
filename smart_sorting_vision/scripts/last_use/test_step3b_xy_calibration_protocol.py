from communication.sorting_protocol import Mode, SortingFrame, SortingProtocol


def main():
    assert int(Mode.CALIB_MOVE) == 0x03
    assert int(Mode.MCU_CALIB_MOVE_ACK) == 0x89
    assert int(Mode.MCU_CALIB_REACHED) == 0x8A
    for mode in (Mode.CALIB_MOVE, Mode.MCU_CALIB_MOVE_ACK, Mode.MCU_CALIB_REACHED):
        f = SortingFrame(mode=mode, x_mm=-72.0, y_mm=5.0)
        raw = SortingProtocol.pack(f)
        d = SortingProtocol.unpack(raw)
        assert d.mode == mode
        assert abs(d.x_mm + 72.0) < 1e-9
        assert abs(d.y_mm - 5.0) < 1e-9
    print("STEP3B XY CALIBRATION PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
