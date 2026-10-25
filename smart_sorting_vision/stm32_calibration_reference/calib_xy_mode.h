#ifndef CALIB_XY_MODE_H
#define CALIB_XY_MODE_H

#include <stdint.h>
#include <stdbool.h>
#include "calib_xy_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Call once after a complete valid 10-byte frame is received. */
void CalibXY_OnFrame(const CalibFrame_t *frame);

/* Call periodically (1~5 ms is suitable) from the main control loop. */
void CalibXY_Task(void);

/* True while calibration mode owns the XY axis. */
bool CalibXY_IsActive(void);

/* ------------------------------------------------------------------
 * Hardware/application adapter functions YOU MUST connect to your
 * existing motion controller and UART driver.
 *
 * Keep the rest of calibration logic unchanged; only implement these.
 * ------------------------------------------------------------------ */
bool RobotXY_StartAbsoluteMoveMm(float x_mm, float y_mm);
bool RobotXY_GetPositionMm(float *x_mm, float *y_mm);
bool RobotXY_IsMotionIdle(void);
void RobotXY_Stop(void);
void JetsonUart_SendBytes(const uint8_t *data, uint16_t len);
uint32_t CalibMillis(void);

#ifdef __cplusplus
}
#endif

#endif
