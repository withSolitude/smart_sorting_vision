#include "dvc_motor_emm_v5.h"

namespace
{
class Interrupt_Guard
{
public:
    Interrupt_Guard() : Previous(__get_PRIMASK()) { __disable_irq(); }
    ~Interrupt_Guard() { __set_PRIMASK(Previous); }
private:
    uint32_t Previous;
};

uint32_t Read_BE32(const uint8_t *p)
{
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

void Record_Ack(Struct_Motor_EmmV5_Ack &ack, uint8_t status, uint32_t now)
{
    ++ack.Count;
    ack.Status = status;
    ack.Timestamp_ms = now;
}
}

void Class_Motor_EmmV5::Init(FDCAN_HandleTypeDef *bus, uint8_t address,
                            uint16_t max_unwrap_gap_ms)
{
    Interrupt_Guard guard;
    Bus = bus;
    Address = address;
    Max_Unwrap_Gap_ms = max_unwrap_gap_ms;
    Have_Unwrap_Sample = false;
    Data = Struct_Motor_EmmV5_Data{};
}

HAL_StatusTypeDef Class_Motor_EmmV5::Send_Position(int64_t signed_pulses,
                                                 uint16_t speed_rpm,
                                                 uint8_t acceleration,
                                                 bool absolute, bool sync)
{
    if (Bus == nullptr || Address == 0U || speed_rpm == 0U ||
        signed_pulses < -INT64_C(0xFFFFFFFF) || signed_pulses > INT64_C(0xFFFFFFFF))
    {
        return HAL_ERROR;
    }
    const uint32_t pulses = static_cast<uint32_t>(signed_pulses < 0 ? -signed_pulses : signed_pulses);
    // 多包命令的每一包都必须在数据首字节重复功能码。
    // 电机地址只放在扩展帧 ID 中，数据区不再重复放地址。
    const uint8_t packet0[8] = {
        0xFD, static_cast<uint8_t>(signed_pulses < 0 ? 1U : 0U),
        static_cast<uint8_t>(speed_rpm >> 8), static_cast<uint8_t>(speed_rpm),
        acceleration, static_cast<uint8_t>(pulses >> 24),
        static_cast<uint8_t>(pulses >> 16), static_cast<uint8_t>(pulses >> 8)};
    const uint8_t packet1[5] = {
        0xFD, static_cast<uint8_t>(pulses), static_cast<uint8_t>(absolute),
        static_cast<uint8_t>(sync), 0x6B};

    Interrupt_Guard guard;
    // 多包命令开始前先预留发送 FIFO 容量。中断调度不能阻塞，
    // FIFO 不足时也不能只发送半条命令。
    if (HAL_FDCAN_GetTxFifoFreeLevel(Bus) < 2U)
    {
        return HAL_BUSY;
    }
    HAL_StatusTypeDef status = CAN_Transmit_Extended_Data(
        Bus, static_cast<uint32_t>(Address) << 8, packet0, sizeof(packet0));
    if (status != HAL_OK) return status;
    return CAN_Transmit_Extended_Data(
        Bus, (static_cast<uint32_t>(Address) << 8) | 1U, packet1, sizeof(packet1));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Enable(bool enable, bool sync)
{
    if (Bus == nullptr || Address == 0U) return HAL_ERROR;
    const uint8_t packet[] = {0xF3, 0xAB, static_cast<uint8_t>(enable),
                              static_cast<uint8_t>(sync), 0x6B};
    return CAN_Transmit_Extended_Data(Bus, static_cast<uint32_t>(Address) << 8,
                                       packet, sizeof(packet));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Stop(bool sync)
{
    if (Bus == nullptr || Address == 0U) return HAL_ERROR;
    const uint8_t packet[] = {0xFE, 0x98, static_cast<uint8_t>(sync), 0x6B};
    return CAN_Transmit_Extended_Data(Bus, static_cast<uint32_t>(Address) << 8,
                                       packet, sizeof(packet));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Sync(FDCAN_HandleTypeDef *bus, uint8_t address)
{
    if (bus == nullptr) return HAL_ERROR;
    const uint8_t packet[] = {0xFF, 0x66, 0x6B};
    return CAN_Transmit_Extended_Data(bus, static_cast<uint32_t>(address) << 8,
                                       packet, sizeof(packet));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Request(uint8_t function)
{
    if (Bus == nullptr || Address == 0U) return HAL_ERROR;
    const uint8_t packet[] = {function, 0x6B};
    return CAN_Transmit_Extended_Data(Bus, static_cast<uint32_t>(Address) << 8,
                                       packet, sizeof(packet));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Request_Encoder() { return Request(0x31); }
HAL_StatusTypeDef Class_Motor_EmmV5::Request_Input_Pulses() { return Request(0x32); }
HAL_StatusTypeDef Class_Motor_EmmV5::Request_Position() { return Request(0x36); }
HAL_StatusTypeDef Class_Motor_EmmV5::Request_Status() { return Request(0x3A); }
HAL_StatusTypeDef Class_Motor_EmmV5::Request_Speed() { return Request(0x35); }

HAL_StatusTypeDef Class_Motor_EmmV5::Trigger_Homing(uint8_t mode, bool sync)
{
    if (Bus == nullptr || Address == 0U || mode > 3U) return HAL_ERROR;
    const uint8_t packet[] = {0x9A, mode, static_cast<uint8_t>(sync), 0x6B};
    return CAN_Transmit_Extended_Data(Bus, static_cast<uint32_t>(Address) << 8,
                                       packet, sizeof(packet));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Stop_Homing()
{
    if (Bus == nullptr || Address == 0U) return HAL_ERROR;
    const uint8_t packet[] = {0x9C, 0x48, 0x6B};
    return CAN_Transmit_Extended_Data(Bus, static_cast<uint32_t>(Address) << 8,
                                       packet, sizeof(packet));
}

HAL_StatusTypeDef Class_Motor_EmmV5::Request_Homing_Status()
{
    return Request(0x3B);
}

HAL_StatusTypeDef Class_Motor_EmmV5::Set_Single_Turn_Origin(bool store)
{
    if (Bus == nullptr || Address == 0U) return HAL_ERROR;
    const uint8_t packet[] = {0x93, 0x88, static_cast<uint8_t>(store), 0x6B};
    return CAN_Transmit_Extended_Data(Bus, static_cast<uint32_t>(Address) << 8,
                                       packet, sizeof(packet));
}

bool Class_Motor_EmmV5::CAN_RxCpltCallback(const FDCAN_RxHeaderTypeDef &header,
                                        const uint8_t *data)
{
    if (Address == 0U || header.IdType != FDCAN_EXTENDED_ID ||
        (header.Identifier >> 8) != Address || (header.Identifier & 0xFFU) != 0U)
    {
        return false;
    }
    // 当前支持的所有应答都能放入一帧经典 CAN 数据帧（第 0 包）。
    const uint32_t length = header.DataLength;
    if (data == nullptr ||
        header.RxFrameType != FDCAN_DATA_FRAME || header.FDFormat != FDCAN_CLASSIC_CAN ||
        length < 3U || length > 8U || data[length - 1U] != 0x6B)
    {
        ++Data.Invalid_Frame_Count;
        return true;
    }
    const uint32_t now = HAL_GetTick();
    switch (data[0])
    {
    case 0x31:
        if (length == 4U)
        {
            const uint16_t encoder = static_cast<uint16_t>((data[1] << 8) | data[2]);
            if (!Have_Unwrap_Sample)
            {
                Have_Unwrap_Sample = true;
                Data.Software_Encoder_Ticks = 0;
                Data.Encoder_Unwrap_Valid = Max_Unwrap_Gap_ms != 0U;
            }
            else if (Data.Encoder_Unwrap_Valid)
            {
                int32_t delta = static_cast<int32_t>(encoder) - Data.Encoder;
                if (delta > 32768) delta -= 65536;
                else if (delta < -32768) delta += 65536;
                if ((now - Data.Encoder_Timestamp_ms) > Max_Unwrap_Gap_ms ||
                    delta == 32768 || delta == -32768)
                {
                    // 反馈间隔过长或恰好半圈时，不能悄悄猜测中间经过了多少圈。
                    Data.Encoder_Unwrap_Valid = false;
                    ++Data.Unwrap_Lost_Count;
                }
                else Data.Software_Encoder_Ticks += delta;
            }
            Data.Encoder = encoder;
            Data.Encoder_Valid = true;
            Data.Encoder_Timestamp_ms = now;
            ++Data.Encoder_Count;
            return true;
        }
        break;
    case 0x32:
    case 0x36:
        if (length == 7U && data[1] <= 1U)
        {
            const int64_t magnitude = Read_BE32(data + 2);
            const int64_t position = data[1] == 0U ? magnitude : -magnitude;
            if (data[0] == 0x36)
            {
                Data.Position_Ticks = position;
                Data.Position_Valid = true;
                Data.Position_Timestamp_ms = now;
                ++Data.Position_Count;
            }
            else
            {
                Data.Input_Pulses = position;
                Data.Input_Pulses_Valid = true;
                Data.Input_Pulses_Timestamp_ms = now;
                ++Data.Input_Pulses_Count;
            }
            return true;
        }
        break;
    case 0x35:
        if (length == 5U && data[1] <= 1U)
        {
            const int32_t magnitude = (static_cast<int32_t>(data[2]) << 8) | data[3];
            Data.Speed_RPM = data[1] == 0U ? magnitude : -magnitude;
            Data.Speed_Valid = true;
            Data.Speed_Timestamp_ms = now;
            ++Data.Speed_Count;
            return true;
        }
        break;
    case 0x3A:
        if (length == 3U)
        {
            Data.Status_Flags = data[1];
            Data.Enabled = (data[1] & 0x01U) != 0U;
            Data.Reached = (data[1] & 0x02U) != 0U;
            Data.Stall = (data[1] & 0x04U) != 0U;
            Data.Stall_Protection = (data[1] & 0x08U) != 0U;
            Data.Status_Valid = true;
            Data.Status_Timestamp_ms = now;
            ++Data.Status_Count;
            return true;
        }
        break;
    case 0x3B:
        if (length == 3U)
        {
            Data.Homing_Flags = data[1];
            Data.Homing_Encoder_Ready = (data[1] & 0x01U) != 0U;
            Data.Homing_Table_Ready = (data[1] & 0x02U) != 0U;
            Data.Homing_In_Progress = (data[1] & 0x04U) != 0U;
            Data.Homing_Failed = (data[1] & 0x08U) != 0U;
            ++Data.Homing_Status_Count;
            return true;
        }
        break;
    case 0xFD:
        if (length == 3U)
        {
            if (data[1] == 0x9FU) ++Data.Reached_Notification_Count;
            else Record_Ack(Data.Position_Ack, data[1], now);
            return true;
        }
        break;
    case 0xF3:
        if (length == 3U) { Record_Ack(Data.Enable_Ack, data[1], now); return true; }
        break;
    case 0xFE:
        if (length == 3U) { Record_Ack(Data.Stop_Ack, data[1], now); return true; }
        break;
    case 0xFF:
        if (length == 3U) { Record_Ack(Data.Sync_Ack, data[1], now); return true; }
        break;
    case 0x9A:
        if (length == 3U) { Record_Ack(Data.Homing_Ack, data[1], now); return true; }
        break;
    case 0x9C:
        if (length == 3U) { Record_Ack(Data.Homing_Stop_Ack, data[1], now); return true; }
        break;
    case 0x93:
        if (length == 3U) { Record_Ack(Data.Origin_Set_Ack, data[1], now); return true; }
        break;
    case 0x00:
        if (length == 3U && data[1] == 0xEEU)
        {
            ++Data.Protocol_Error_Count;
            Data.Protocol_Error_Timestamp_ms = now;
            return true;
        }
        break;
    default:
        break;
    }
    ++Data.Invalid_Frame_Count;
    return true;
}

Struct_Motor_EmmV5_Data Class_Motor_EmmV5::Get_Data() const
{
    Interrupt_Guard guard;
    return Data;
}

void Class_Motor_EmmV5::Reset_Encoder_Unwrap()
{
    Interrupt_Guard guard;
    Have_Unwrap_Sample = false;
    Data.Encoder_Unwrap_Valid = false;
    Data.Software_Encoder_Ticks = 0;
}
