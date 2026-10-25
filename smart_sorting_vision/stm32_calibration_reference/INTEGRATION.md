# STM32 XY calibration mode integration

This folder is a reference module because the STM32 motion-control source was not included in the Jetson archive.

## New modes

- Jetson -> STM32: `0x03 CALIB_MOVE`, X/Y use the existing signed int16, 0.1 mm/LSB encoding.
- STM32 -> Jetson: `0x89 MCU_CALIB_MOVE_ACK` after accepting the target.
- STM32 -> Jetson: `0x8A MCU_CALIB_REACHED` only after measured XY is within 0.30 mm and motion is idle for 150 ms.
- Errors reuse `0xE0 MCU_ERROR`; `box_id` is the calibration error code.

## Required lower-controller adapters

Implement these functions in your existing motion project:

```c
bool RobotXY_StartAbsoluteMoveMm(float x_mm, float y_mm);
bool RobotXY_GetPositionMm(float *x_mm, float *y_mm);
bool RobotXY_IsMotionIdle(void);
void RobotXY_Stop(void);
void JetsonUart_SendBytes(const uint8_t *data, uint16_t len);
uint32_t CalibMillis(void);
```

Do not make `RobotXY_StartAbsoluteMoveMm` execute the normal pick sequence. Calibration mode must own XY only.

After your normal 10-byte UART parser decodes a frame, call `CalibXY_OnFrame(&frame)`. Call `CalibXY_Task()` in the regular 1-5 ms control loop.
