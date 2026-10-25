#ifndef TEST_DM_H
#define TEST_DM_H
#include "test_hardware.h"
enum Enum_Motor_DM_Status { Motor_DM_Status_DISABLE, Motor_DM_Status_ENABLE };
enum Enum_Motor_DM_Control_Status { Motor_DM_Control_Status_DISABLE, Motor_DM_Control_Status_ENABLE };
enum Enum_Motor_DM_Control_Method { Motor_DM_Control_Method_NORMAL_MIT };
class Class_Motor_DM_Normal
{
public:
    Enum_Motor_DM_Status Status=Motor_DM_Status_ENABLE;
    Enum_Motor_DM_Control_Status Control_Status=Motor_DM_Control_Status_ENABLE;
    bool Position_Valid=true;
    float Angle=0, Omega=0, Torque=0;
    uint32_t Position_Count=0, Feedback_Count=0, Feedback_Tick=0;
    uint32_t Rx_Calls=0, Read_Calls=0, Send_Calls=0;
    uint32_t Alive_Calls=0, Clear_Error_Calls=0, Enter_Calls=0;
    bool Alive_Auto_Enable=false;
    template<class... Args> void Init(Args...) {}
    void CAN_Send_Clear_Error() { ++Clear_Error_Calls; }
    void CAN_Send_Enter()
    {
        ++Enter_Calls;
        if(Alive_Auto_Enable) {Status=Motor_DM_Status_ENABLE;Control_Status=Motor_DM_Control_Status_ENABLE;Refresh();}
    }
    void CAN_Send_Read_Output_Shaft_Angle() { ++Read_Calls; }
    void CAN_RxCpltCallback() { ++Rx_Calls; }
    void TIM_Alive_PeriodElapsedCallback()
    {
        ++Alive_Calls;
        // Model the real DM driver: Alive may actively clear faults and
        // re-enable its drive. Calling it is not a passive status observation.
        if(Alive_Auto_Enable && (Status!=Motor_DM_Status_ENABLE || Control_Status!=Motor_DM_Control_Status_ENABLE))
        { CAN_Send_Clear_Error();CAN_Send_Enter(); }
    }
    void TIM_Send_PeriodElapsedCallback() { ++Send_Calls; }
    Enum_Motor_DM_Status Get_Status() const { return Status; }
    Enum_Motor_DM_Control_Status Get_Control_Status() const { return Control_Status; }
    bool Get_Output_Shaft_Angle_Valid() const { return Position_Valid; }
    float Get_Output_Shaft_Angle() const { return Angle; }
    float Get_Now_Angle() const { return Angle; }
    float Get_Now_Omega() const { return Omega; }
    uint32_t Get_Output_Shaft_Update_Count() const { return Position_Count; }
    uint32_t Get_Feedback_Update_Count() const { return Feedback_Count; }
    uint32_t Get_Feedback_Timestamp_ms() const { return Feedback_Tick; }
    bool Get_Feedback_Valid() const { return Feedback_Count != 0U; }
    void Set_Control_Angle(float) {}
    void Set_Control_Omega(float) {}
    void Set_K_P(float) {}
    void Set_K_D(float) {}
    void Set_Control_Torque(float value) { Torque=value; }
    void Refresh() { ++Position_Count; ++Feedback_Count; Feedback_Tick=HAL_GetTick(); }
};
#endif
