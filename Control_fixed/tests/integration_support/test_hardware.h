#ifndef TEST_HARDWARE_H
#define TEST_HARDWARE_H
#include "emm_v5_hal_stub.h"
#include <vector>

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t HAL_UART_STATE_READY = 0;
constexpr uint32_t HAL_UART_STATE_BUSY_TX = 1;
struct UART_HandleTypeDef { uint32_t gState = HAL_UART_STATE_READY; };
extern UART_HandleTypeDef huart3, huart4;
void HAL_Delay(uint32_t ms);
#endif
