#ifndef CALIB_XY_PROTOCOL_H
#define CALIB_XY_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SORT_FRAME_LEN 10u
#define SORT_HEADER0   0xA5u
#define SORT_HEADER1   0x5Au
#define SORT_TAIL0     0x0Du
#define SORT_TAIL1     0x0Au

#define MODE_CALIB_MOVE          0x03u
#define MODE_STOP                0x7Fu
#define MODE_MCU_CALIB_MOVE_ACK  0x89u
#define MODE_MCU_CALIB_REACHED   0x8Au
#define MODE_MCU_ERROR           0xE0u

/* 0.1 mm / LSB; signed int16, little endian. */
typedef struct
{
    uint8_t mode;
    float x_mm;
    float y_mm;
    uint8_t box_id;
} CalibFrame_t;

bool CalibProtocol_Decode10(const uint8_t raw[SORT_FRAME_LEN], CalibFrame_t *out);
void CalibProtocol_Encode10(uint8_t raw[SORT_FRAME_LEN], uint8_t mode,
                            float x_mm, float y_mm, uint8_t box_id);

#ifdef __cplusplus
}
#endif

#endif
