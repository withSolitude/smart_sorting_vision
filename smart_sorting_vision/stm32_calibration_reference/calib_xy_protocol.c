#include "calib_xy_protocol.h"
#include <math.h>

static int16_t mm_to_raw(float mm)
{
    float scaled = mm * 10.0f;
    if (scaled >= 0.0f)
        return (int16_t)(scaled + 0.5f);
    return (int16_t)(scaled - 0.5f);
}

bool CalibProtocol_Decode10(const uint8_t raw[SORT_FRAME_LEN], CalibFrame_t *out)
{
    int16_t x_raw;
    int16_t y_raw;

    if (raw == 0 || out == 0) return false;
    if (raw[0] != SORT_HEADER0 || raw[1] != SORT_HEADER1) return false;
    if (raw[8] != SORT_TAIL0 || raw[9] != SORT_TAIL1) return false;

    x_raw = (int16_t)((uint16_t)raw[3] | ((uint16_t)raw[4] << 8));
    y_raw = (int16_t)((uint16_t)raw[5] | ((uint16_t)raw[6] << 8));

    out->mode = raw[2];
    out->x_mm = ((float)x_raw) / 10.0f;
    out->y_mm = ((float)y_raw) / 10.0f;
    out->box_id = raw[7];
    return true;
}

void CalibProtocol_Encode10(uint8_t raw[SORT_FRAME_LEN], uint8_t mode,
                            float x_mm, float y_mm, uint8_t box_id)
{
    int16_t x_raw = mm_to_raw(x_mm);
    int16_t y_raw = mm_to_raw(y_mm);

    raw[0] = SORT_HEADER0;
    raw[1] = SORT_HEADER1;
    raw[2] = mode;
    raw[3] = (uint8_t)((uint16_t)x_raw & 0xFFu);
    raw[4] = (uint8_t)(((uint16_t)x_raw >> 8) & 0xFFu);
    raw[5] = (uint8_t)((uint16_t)y_raw & 0xFFu);
    raw[6] = (uint8_t)(((uint16_t)y_raw >> 8) & 0xFFu);
    raw[7] = box_id;
    raw[8] = SORT_TAIL0;
    raw[9] = SORT_TAIL1;
}
