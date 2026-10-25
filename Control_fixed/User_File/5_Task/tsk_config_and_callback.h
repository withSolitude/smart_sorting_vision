#ifndef TEST_ROBOWAKER_TSK_CONFIG_AND_CALLBACK_H
#define TEST_ROBOWAKER_TSK_CONFIG_AND_CALLBACK_H

#include "stm32h7xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

    void Task_Init();
    void Task_Loop();

    // Manual origin confirmation from debugger or a dedicated operator UI.
    // Place the carriage at this known coordinate, write X/Y, then Request=1.
    extern volatile uint8_t XY_Origin_Confirm_Request;
    extern volatile float XY_Origin_Confirm_X_mm;
    extern volatile float XY_Origin_Confirm_Y_mm;
    extern volatile HAL_StatusTypeDef XY_Origin_Confirm_Status;
    // 置1后由前台任务请求一次自动碰撞回零；完成/失败后自动清零。
    extern volatile uint8_t XY_Collision_Homing_Request;
    extern volatile HAL_StatusTypeDef XY_Collision_Homing_Status;

#ifdef __cplusplus
};
#endif

#endif //TEST_ROBOWAKER_TSK_CONFIG_AND_CALLBACK_H

