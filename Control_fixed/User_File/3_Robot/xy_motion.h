#ifndef XY_MOTION_H
#define XY_MOTION_H

#include "3_Robot/xy_motion_config.h"
#include "2_Device/Motor/Motor_EmmV5/dvc_motor_emm_v5.h"

enum class XY_Fault : uint8_t
{
    None, Configuration, Feedback_Timeout, Drive_Stall, Drive_Disabled,
    Command_Rejected, Command_Timeout, Transport, Move_Timeout, Position_Jump,
    Homing_Failed
};

class Class_XY_Motion
{
public:
    Class_Motor_EmmV5 Motor_A, Motor_B;
    void Init();
    void TIM_1ms_PeriodElapsedCallback();
    void CAN_RxCpltCallback(const FDCAN_RxHeaderTypeDef &header, const uint8_t *data);
    HAL_StatusTypeDef Confirm_Origin_mm(float x, float y);
    HAL_StatusTypeDef Move_To_mm(float x, float y, float vx, float vy);
    void Stop();
    bool Get_Position_Valid() const;
    bool Get_Fault() const { return Fault != XY_Fault::None; }
    XY_Fault Get_Fault_Code() const { return Fault; }
    bool Get_Motion_Finished() const;
    bool Get_Stop_Confirmed() const;
    // 启动安全的“单电机、单轴、顺序碰撞回零”。默认配置关闭。
    HAL_StatusTypeDef Start_Collision_Homing();
    void Stop_Collision_Homing();
    bool Get_Homing_Active() const { return Homing_Active; }
    bool Get_Homing_Failed() const { return Homing_Failed; }
    float Get_X_mm() const { return static_cast<float>(Now_X); }
    float Get_Y_mm() const { return static_cast<float>(Now_Y); }
    static void XY_To_Motor_mm(double x, double y, double &a, double &b);
    static void Motor_To_XY_mm(double a, double b, double &x, double &y);

private:
    enum class State : uint8_t
    {
        Idle, Enable_A, Wait_Enable_A, Enable_B, Wait_Enable_B,
        Arm_A, Wait_Arm_A, Arm_B, Wait_Arm_B, Sync, Wait_Sync, Moving,
        Stop_A, Stop_B, Stopping,
        Home_Enable_A, Home_Wait_Enable_A, Home_Enable_B, Home_Wait_Enable_B,
        Home_B, Home_Wait_B, Home_Hold_B, Home_Wait_Hold_B,
        Home_A, Home_Wait_A, Home_Hold_A, Home_Wait_Hold_A, Home_Finish
    };
    State Phase = State::Idle;
    XY_Fault Fault = XY_Fault::None;
    bool Origin_Valid = false;
    double Origin_X = 0, Origin_Y = 0, Now_X = 0, Now_Y = 0;
    int64_t Origin_A = 0, Origin_B = 0, Target_A = 0, Target_B = 0;
    int64_t Last_A = 0, Last_B = 0;
    uint16_t Speed_A = 1, Speed_B = 1;
    uint32_t Phase_Tick = 0, Move_Tick = 0, Poll_Tick = 0;
    uint32_t Stable_Tick = 0, Ack_Baseline = 0;
    uint32_t Position_A_Baseline = 0, Position_B_Baseline = 0;
    uint32_t Status_A_Baseline = 0, Status_B_Baseline = 0;
    uint32_t Speed_A_Baseline = 0, Speed_B_Baseline = 0;
    uint32_t Last_Position_A_Count = 0, Last_Position_B_Count = 0;
    uint32_t Last_Position_A_Tick = 0, Last_Position_B_Tick = 0;
    uint32_t Stop_A_Baseline = 0, Stop_B_Baseline = 0;
    uint8_t Poll_Slot = 0;
    bool Stable = false;
    bool Track_Enabled = false;
    bool Motion_Completed = false;
    // 只有真实完成双轴STOP应答和停稳反馈后才置位；防止重复STOP重启停车流程。
    bool Stop_Sequence_Confirmed = false;
    bool Homing_Active = false;
    bool Homing_Failed = false;
    uint32_t Homing_Tick = 0, Homing_Status_Tick = 0;
    uint32_t Homing_Status_A_Baseline = 0, Homing_Status_B_Baseline = 0;
    uint32_t Homing_Ack_Baseline = 0;
    int64_t Homing_B_Position_Ticks = 0;
    int64_t Homing_A_Position_Ticks = 0;
    int64_t Homing_Hold_Pulses = 0;
    bool Homing_Stop_Requested = false;
    uint32_t Protocol_Error_A_Baseline = 0, Protocol_Error_B_Baseline = 0;
    uint32_t Stable_Position_A_Count = 0, Stable_Position_B_Count = 0;
    bool Config_Valid() const;
    bool Feedback_Fresh() const;
    void Set_Phase(State phase);
    void Set_Fault(XY_Fault fault);
    void Begin_Stop();
    void Fail_Homing();
    void Poll();
    void Update_Position();
    void Capture_Feedback_Baselines();
};

#endif
