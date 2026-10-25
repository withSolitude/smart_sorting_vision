#ifndef TEST_SERVO_H
#define TEST_SERVO_H
#include "test_hardware.h"
enum Enum_Servo_Motor_Work_Mode { Servo_Motor_Work_Mode_270_CW };
class Class_Servo_Motor
{
public:
    struct Command { float Value; uint16_t Duration; uint32_t Tick; bool PWM; };
    UART_HandleTypeDef *Port = nullptr;
    uint8_t ID = 0;
    uint8_t Forced_Status = HAL_OK;
    uint32_t Stop_Attempts = 0, Stops_Accepted = 0;
    std::vector<Command> Commands;
    float Target_Angle = 0;
    uint16_t Target_PWM = 0;
    void Init(UART_HandleTypeDef *port, uint8_t id, Enum_Servo_Motor_Work_Mode)
    { Port=port; ID=id; }
    uint8_t Control_Angle(float value, uint16_t ms)
    {
        const uint8_t result = Ready();
        if (result == HAL_OK) { Target_Angle=value; Commands.push_back({value,ms,HAL_GetTick(),false}); }
        return result;
    }
    uint8_t Control_PWM(uint16_t value, uint16_t ms)
    {
        const uint8_t result = Ready();
        if (result == HAL_OK) { Target_PWM=value; Commands.push_back({static_cast<float>(value),ms,HAL_GetTick(),true}); }
        return result;
    }
    uint8_t Stop() { ++Stop_Attempts; const uint8_t r=Ready(); if(r==HAL_OK) ++Stops_Accepted; return r; }
    void Set_Target_PWM(uint16_t v) { Target_PWM=v; }
    float Get_Target_Angle() const { return Target_Angle; }
    uint16_t Get_Target_PWM() const { return Target_PWM; }
    void UART_RxCpltCallback(uint8_t*,uint16_t) {}
private:
    uint8_t Ready() const { return Port && Port->gState != HAL_UART_STATE_READY ? static_cast<uint8_t>(HAL_BUSY) : Forced_Status; }
};
#endif
