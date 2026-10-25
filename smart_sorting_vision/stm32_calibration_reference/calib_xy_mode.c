#include "calib_xy_mode.h"
#include <math.h>

#define CALIB_X_MIN_MM        (-77.0f)
#define CALIB_X_MAX_MM        (  0.0f)
#define CALIB_Y_MIN_MM        (  0.0f)
#define CALIB_Y_MAX_MM        ( 55.0f)

/* Position must stay inside this window continuously before REACHED is sent. */
#define CALIB_REACHED_TOL_MM  (0.30f)
#define CALIB_STABLE_MS        150u
#define CALIB_MOVE_TIMEOUT_MS 15000u

/* Error codes are returned in frame.box_id. */
#define CALIB_ERR_OUT_OF_RANGE  1u
#define CALIB_ERR_BUSY          2u
#define CALIB_ERR_START_FAILED  3u
#define CALIB_ERR_TIMEOUT       4u
#define CALIB_ERR_POSITION      5u

static bool s_active = false;
static bool s_waiting_reached = false;
static float s_target_x = 0.0f;
static float s_target_y = 0.0f;
static uint32_t s_move_start_ms = 0u;
static uint32_t s_in_tolerance_since_ms = 0u;

static bool inside_workspace(float x, float y)
{
    return (x >= CALIB_X_MIN_MM && x <= CALIB_X_MAX_MM &&
            y >= CALIB_Y_MIN_MM && y <= CALIB_Y_MAX_MM);
}

static void send_frame(uint8_t mode, float x_mm, float y_mm, uint8_t code)
{
    uint8_t raw[SORT_FRAME_LEN];
    CalibProtocol_Encode10(raw, mode, x_mm, y_mm, code);
    JetsonUart_SendBytes(raw, SORT_FRAME_LEN);
}

static void send_error(uint8_t code)
{
    send_frame(MODE_MCU_ERROR, s_target_x, s_target_y, code);
}

void CalibXY_OnFrame(const CalibFrame_t *frame)
{
    if (frame == 0) return;

    if (frame->mode == MODE_STOP)
    {
        RobotXY_Stop();
        s_active = false;
        s_waiting_reached = false;
        s_in_tolerance_since_ms = 0u;
        return;
    }

    if (frame->mode != MODE_CALIB_MOVE)
        return;

    s_target_x = frame->x_mm;
    s_target_y = frame->y_mm;

    if (!inside_workspace(s_target_x, s_target_y))
    {
        send_error(CALIB_ERR_OUT_OF_RANGE);
        return;
    }

    /* Reject a new calibration target while the old target is still moving. */
    if (s_waiting_reached || !RobotXY_IsMotionIdle())
    {
        send_error(CALIB_ERR_BUSY);
        return;
    }

    /* IMPORTANT: this adapter must command XY only; no gripper/pick/sort action. */
    if (!RobotXY_StartAbsoluteMoveMm(s_target_x, s_target_y))
    {
        send_error(CALIB_ERR_START_FAILED);
        return;
    }

    s_active = true;
    s_waiting_reached = true;
    s_move_start_ms = CalibMillis();
    s_in_tolerance_since_ms = 0u;

    /* Echo accepted coordinate back to Jetson. */
    send_frame(MODE_MCU_CALIB_MOVE_ACK, s_target_x, s_target_y, 0u);
}

void CalibXY_Task(void)
{
    float x_now, y_now;
    float dx, dy;
    uint32_t now;

    if (!s_waiting_reached)
        return;

    now = CalibMillis();

    if ((uint32_t)(now - s_move_start_ms) > CALIB_MOVE_TIMEOUT_MS)
    {
        RobotXY_Stop();
        send_error(CALIB_ERR_TIMEOUT);
        s_waiting_reached = false;
        s_active = false;
        s_in_tolerance_since_ms = 0u;
        return;
    }

    if (!RobotXY_GetPositionMm(&x_now, &y_now))
    {
        send_error(CALIB_ERR_POSITION);
        s_waiting_reached = false;
        s_active = false;
        s_in_tolerance_since_ms = 0u;
        return;
    }

    dx = fabsf(x_now - s_target_x);
    dy = fabsf(y_now - s_target_y);

    if (dx <= CALIB_REACHED_TOL_MM && dy <= CALIB_REACHED_TOL_MM && RobotXY_IsMotionIdle())
    {
        if (s_in_tolerance_since_ms == 0u)
            s_in_tolerance_since_ms = now;

        if ((uint32_t)(now - s_in_tolerance_since_ms) >= CALIB_STABLE_MS)
        {
            /* Return measured current position, not only the target. */
            send_frame(MODE_MCU_CALIB_REACHED, x_now, y_now, 0u);
            s_waiting_reached = false;
            s_active = false;
            s_in_tolerance_since_ms = 0u;
        }
    }
    else
    {
        s_in_tolerance_since_ms = 0u;
    }
}

bool CalibXY_IsActive(void)
{
    return s_active;
}
