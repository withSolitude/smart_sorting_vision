#ifndef DVC_MOTOR_EMM_V5_H
#define DVC_MOTOR_EMM_V5_H

#include <cstdint>
#ifdef EMM_V5_HOST_TEST
#include "emm_v5_hal_stub.h"
#else
#include "1_Middleware/1_Driver/CAN/drv_can.h"
#endif

// Emm42 V5 协议：经典 CAN 扩展帧，校验字节固定为 0x6B。
// 有符号位置为正表示顺时针（CW），为负表示逆时针（CCW）。
struct Struct_Motor_EmmV5_Ack
{
    uint32_t Count = 0;
    uint32_t Timestamp_ms = 0;
    uint8_t Status = 0;
};

struct Struct_Motor_EmmV5_Data
{
    // 0x36 是驱动器提供的有符号多圈位置反馈，每圈 65536 个编码器刻度。
    int64_t Position_Ticks = 0;
    // 0x31 在主控侧独立展开多圈，数值相对第一次采样点计算。
    int64_t Software_Encoder_Ticks = 0;
    int64_t Input_Pulses = 0;
    uint16_t Encoder = 0;
    int32_t Speed_RPM = 0;
    uint8_t Status_Flags = 0;
    // 0x3B：bit0编码器就绪，bit1校准表就绪，bit2正在回零，bit3回零失败。
    uint8_t Homing_Flags = 0;
    bool Position_Valid = false;
    bool Encoder_Valid = false;
    bool Encoder_Unwrap_Valid = false;
    bool Input_Pulses_Valid = false;
    bool Speed_Valid = false;
    bool Status_Valid = false;
    bool Enabled = false;
    bool Reached = false;
    bool Stall = false;
    bool Stall_Protection = false;
    bool Homing_Encoder_Ready = false;
    bool Homing_Table_Ready = false;
    bool Homing_In_Progress = false;
    bool Homing_Failed = false;
    uint32_t Position_Timestamp_ms = 0;
    uint32_t Encoder_Timestamp_ms = 0;
    uint32_t Speed_Timestamp_ms = 0;
    uint32_t Status_Timestamp_ms = 0;
    uint32_t Input_Pulses_Timestamp_ms = 0;
    uint32_t Position_Count = 0;
    uint32_t Encoder_Count = 0;
    uint32_t Speed_Count = 0;
    uint32_t Status_Count = 0;
    uint32_t Homing_Status_Count = 0;
    uint32_t Input_Pulses_Count = 0;
    uint32_t Invalid_Frame_Count = 0;
    uint32_t Unwrap_Lost_Count = 0;
    uint32_t Protocol_Error_Count = 0;
    uint32_t Protocol_Error_Timestamp_ms = 0;
    uint32_t Reached_Notification_Count = 0;
    Struct_Motor_EmmV5_Ack Position_Ack;
    Struct_Motor_EmmV5_Ack Enable_Ack;
    Struct_Motor_EmmV5_Ack Stop_Ack;
    Struct_Motor_EmmV5_Ack Sync_Ack;
    Struct_Motor_EmmV5_Ack Homing_Ack;
    Struct_Motor_EmmV5_Ack Homing_Stop_Ack;
    Struct_Motor_EmmV5_Ack Origin_Set_Ack;
};

class Class_Motor_EmmV5
{
public:
    // max_unwrap_gap_ms 必须严格小于 30000/max_RPM。展开失效后，
    // 0x31 在显式复位前保持无效；0x36 仍可继续使用。
    void Init(FDCAN_HandleTypeDef *bus, uint8_t address,
              uint16_t max_unwrap_gap_ms = 5);
    HAL_StatusTypeDef Send_Position(int64_t signed_pulses, uint16_t speed_rpm,
                                    uint8_t acceleration, bool absolute = true,
                                    bool sync = true);
    HAL_StatusTypeDef Enable(bool enable, bool sync = false);
    HAL_StatusTypeDef Stop(bool sync = false);
    static HAL_StatusTypeDef Sync(FDCAN_HandleTypeDef *bus, uint8_t address = 0);
    HAL_StatusTypeDef Request_Encoder(); // 读取 0x31：校准后的单圈编码器
    HAL_StatusTypeDef Request_Input_Pulses(); // 读取 0x32：输入脉冲诊断
    HAL_StatusTypeDef Request_Position(); // 读取 0x36：实测多圈位置
    HAL_StatusTypeDef Request_Status(); // 读取 0x3A：状态标志
    HAL_StatusTypeDef Request_Speed(); // 读取 0x35：实时速度
    // 触发回零：0=单圈就近，1=单圈方向，2=多圈碰撞，3=限位开关。
    HAL_StatusTypeDef Trigger_Homing(uint8_t mode = 2U, bool sync = false);
    // 强制中断回零。
    HAL_StatusTypeDef Stop_Homing();
    // 读取 0x3B 回零状态标志。
    HAL_StatusTypeDef Request_Homing_Status();
    // 设置单圈回零零点；多圈碰撞回零完成后默认不调用。
    HAL_StatusTypeDef Set_Single_Turn_Origin(bool store = true);
    bool CAN_RxCpltCallback(const FDCAN_RxHeaderTypeDef &header,
                           const uint8_t *data);
    // 在很短的关中断临界区内复制数据，保证 64 位位置读取不会被打断。
    Struct_Motor_EmmV5_Data Get_Data() const;
    void Reset_Encoder_Unwrap();
    uint8_t Get_Address() const { return Address; }

private:
    FDCAN_HandleTypeDef *Bus = nullptr;
    uint8_t Address = 0;
    uint16_t Max_Unwrap_Gap_ms = 5;
    bool Have_Unwrap_Sample = false;
    Struct_Motor_EmmV5_Data Data;
    HAL_StatusTypeDef Request(uint8_t function);
};

#endif
