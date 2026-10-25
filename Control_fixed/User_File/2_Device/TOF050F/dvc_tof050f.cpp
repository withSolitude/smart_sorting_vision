//
// Created by Lenovo on 2026/9/2.
//

#include "2_Device/TOF050F/dvc_tof050f.h"


// ============================================================
// TOF050F寄存器
// ============================================================

static constexpr uint16_t
TOF050F_REG_SPECIAL =
    0x0001;


static constexpr uint16_t
TOF050F_REG_DEVICE_ADDRESS =
    0x0002;


static constexpr uint16_t
TOF050F_REG_BAUDRATE =
    0x0003;


static constexpr uint16_t
TOF050F_REG_RANGE_MODE =
    0x0004;


static constexpr uint16_t
TOF050F_REG_AUTO_OUTPUT =
    0x0005;


static constexpr uint16_t
TOF050F_REG_LOAD_CALIBRATION =
    0x0006;


static constexpr uint16_t
TOF050F_REG_OFFSET =
    0x0007;


static constexpr uint16_t
TOF050F_REG_XTALK =
    0x0008;


static constexpr uint16_t
TOF050F_REG_IIC_MODE =
    0x0009;


// ============================================================
// 测距结果
// ============================================================

static constexpr uint16_t
TOF050F_REG_DISTANCE =
    0x0010;


// ============================================================
// 特殊命令
// ============================================================

static constexpr uint16_t
TOF050F_SPECIAL_RESTORE_DEFAULT =
    0xAA55;


static constexpr uint16_t
TOF050F_SPECIAL_RESTART =
    0x1000;


// ============================================================
// Init
// ============================================================

void Class_TOF050F::Init(
    UART_HandleTypeDef *huart,
    uint8_t Address)
{
    UART_Handler =
        huart;


    Device_Address =
        Address;


    Status =
        TOF050F_Status_ENABLE;


    Data.Distance_mm =
        0;


    Data.Distance_Valid =
        false;


    Data.Online =
        false;


    Data.Rx_Count =
        0;


    Data.CRC_Error_Count =
        0;


    Data.Frame_Error_Count =
        0;


    Data.Timeout_Count =
        0;


    Waiting_Response =
        false;


    Response_Timeout_Count =
        0;


    Offline_Count =
        0;


    Auto_Request_Enable =
        true;


    Pending_Address_Change =
        false;


    Pending_New_Address =
        Device_Address;
}


// ============================================================
// CRC-16 / MODBUS
//
// Polynomial:
// 0xA001
//
// Initial:
// 0xFFFF
// ============================================================

uint16_t Class_TOF050F::Calculate_CRC16(
    const uint8_t *Data_Buffer,
    uint16_t Length) const
{
    uint16_t crc =
        0xFFFF;


    for (uint16_t i = 0;
         i < Length;
         i++)
    {
        crc ^=
            Data_Buffer[i];


        for (uint8_t j = 0;
             j < 8;
             j++)
        {
            if (crc & 0x0001)
            {
                crc >>= 1;

                crc ^=
                    0xA001;
            }
            else
            {
                crc >>= 1;
            }
        }
    }


    return crc;
}


// ============================================================
// CRC检查
// ============================================================

bool Class_TOF050F::Check_CRC(
    const uint8_t *Buffer,
    uint16_t Length) const
{
    if (Buffer == nullptr ||
        Length < 4)
    {
        return false;
    }


    const uint16_t calculated_crc =
        Calculate_CRC16(
            Buffer,
            Length - 2
        );


    // ========================================================
    // Modbus：
    //
    // CRC低位在前
    // CRC高位在后
    // ========================================================

    const uint16_t received_crc =
        static_cast<uint16_t>(
            Buffer[Length - 2]
        )
        |
        (
            static_cast<uint16_t>(
                Buffer[Length - 1]
            )
            << 8
        );


    return
        calculated_crc ==
        received_crc;
}


// ============================================================
// Transmit Frame
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Transmit_Frame()
{
    if (UART_Handler == nullptr)
    {
        return HAL_ERROR;
    }


    // ========================================================
    // 上一次DMA还没有发送完成
    //
    // 这里不要修改Tx_Buffer
    // ========================================================

    if (UART_Handler->gState !=
        HAL_UART_STATE_READY)
    {
        return HAL_BUSY;
    }


    HAL_StatusTypeDef status =
        HAL_UART_Transmit_DMA(
            UART_Handler,
            Tx_Buffer,
            sizeof(Tx_Buffer)
        );


    if (status == HAL_OK)
    {
        Waiting_Response =
            true;


        Response_Timeout_Count =
            0;
    }


    return status;
}


// ============================================================
// Read Holding Register
//
// Modbus Function 03
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Read_Holding_Register(
    uint16_t Register_Address,
    uint16_t Register_Number)
{
    if (UART_Handler == nullptr)
    {
        return HAL_ERROR;
    }


    // ========================================================
    // 如果UART还在发送
    // 绝对不能改Tx_Buffer
    // ========================================================

    if (UART_Handler->gState !=
        HAL_UART_STATE_READY)
    {
        return HAL_BUSY;
    }


    Tx_Buffer[0] =
        Device_Address;


    Tx_Buffer[1] =
        0x03;


    Tx_Buffer[2] =
        static_cast<uint8_t>(
            Register_Address >> 8
        );


    Tx_Buffer[3] =
        static_cast<uint8_t>(
            Register_Address & 0xFF
        );


    Tx_Buffer[4] =
        static_cast<uint8_t>(
            Register_Number >> 8
        );


    Tx_Buffer[5] =
        static_cast<uint8_t>(
            Register_Number & 0xFF
        );


    const uint16_t crc =
        Calculate_CRC16(
            Tx_Buffer,
            6
        );


    // CRC低位
    Tx_Buffer[6] =
        static_cast<uint8_t>(
            crc & 0xFF
        );


    // CRC高位
    Tx_Buffer[7] =
        static_cast<uint8_t>(
            crc >> 8
        );


    return
        Transmit_Frame();
}


// ============================================================
// Write Single Register
//
// Modbus Function 06
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Write_Single_Register(
    uint16_t Register_Address,
    uint16_t Register_Data)
{
    if (UART_Handler == nullptr)
    {
        return HAL_ERROR;
    }


    if (UART_Handler->gState !=
        HAL_UART_STATE_READY)
    {
        return HAL_BUSY;
    }


    Tx_Buffer[0] =
        Device_Address;


    Tx_Buffer[1] =
        0x06;


    Tx_Buffer[2] =
        static_cast<uint8_t>(
            Register_Address >> 8
        );


    Tx_Buffer[3] =
        static_cast<uint8_t>(
            Register_Address & 0xFF
        );


    Tx_Buffer[4] =
        static_cast<uint8_t>(
            Register_Data >> 8
        );


    Tx_Buffer[5] =
        static_cast<uint8_t>(
            Register_Data & 0xFF
        );


    const uint16_t crc =
        Calculate_CRC16(
            Tx_Buffer,
            6
        );


    Tx_Buffer[6] =
        static_cast<uint8_t>(
            crc & 0xFF
        );


    Tx_Buffer[7] =
        static_cast<uint8_t>(
            crc >> 8
        );


    return
        Transmit_Frame();
}


// ============================================================
// Request Distance
//
// 读取：
// 0x0010
//
// 数量：
// 1
//
// 地址1时理论发送：
//
// 01 03 00 10 00 01 85 CF
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Request_Distance()
{
    if (Status !=
        TOF050F_Status_ENABLE)
    {
        return HAL_ERROR;
    }


    // ========================================================
    // 上一次请求还没有回来
    // 暂时不重新发送
    // ========================================================

    if (Waiting_Response)
    {
        return HAL_BUSY;
    }


    return
        Read_Holding_Register(
            TOF050F_REG_DISTANCE,
            1
        );
}


// ============================================================
// Set Range Mode
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Set_Range_Mode(
    Enum_TOF050F_Range_Mode Range_Mode)
{
    return
        Write_Single_Register(
            TOF050F_REG_RANGE_MODE,
            static_cast<uint16_t>(
                Range_Mode
            )
        );
}


// ============================================================
// Set Auto Output Period
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Set_Auto_Output_Period(
    uint16_t Period_ms)
{
    return
        Write_Single_Register(
            TOF050F_REG_AUTO_OUTPUT,
            Period_ms
        );
}


// ============================================================
// Set Load Calibration
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Set_Load_Calibration(
    bool Enable)
{
    return
        Write_Single_Register(
            TOF050F_REG_LOAD_CALIBRATION,
            Enable ?
                0x0001 :
                0x0000
        );
}


// ============================================================
// Set Device Address
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Set_Device_Address(
    uint8_t New_Address)
{
    // 0为广播地址
    // 一般不作为普通设备地址使用

    if (New_Address == 0)
    {
        return HAL_ERROR;
    }


    Pending_Address_Change =
        true;


    Pending_New_Address =
        New_Address;


    HAL_StatusTypeDef status =
        Write_Single_Register(
            TOF050F_REG_DEVICE_ADDRESS,
            New_Address
        );


    if (status != HAL_OK)
    {
        Pending_Address_Change =
            false;
    }


    return status;
}


// ============================================================
// Restart
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Restart()
{
    return
        Write_Single_Register(
            TOF050F_REG_SPECIAL,
            TOF050F_SPECIAL_RESTART
        );
}


// ============================================================
// Restore Default
// ============================================================

HAL_StatusTypeDef
Class_TOF050F::Restore_Default()
{
    return
        Write_Single_Register(
            TOF050F_REG_SPECIAL,
            TOF050F_SPECIAL_RESTORE_DEFAULT
        );
}


// ============================================================
// Parse Read Response
//
// 正常测距回复：
//
// Address
// 03
// 02
// Distance_H
// Distance_L
// CRC_L
// CRC_H
//
// 共7字节
// ============================================================

bool Class_TOF050F::Parse_Read_Response(
    const uint8_t *Buffer,
    uint16_t Length)
{
    if (Length < 7)
    {
        return false;
    }


    // ========================================================
    // 扫描Buffer
    //
    // 防止UART Idle DMA一次回调中
    // 前面包含其他无效数据
    // ========================================================

    for (uint16_t i = 0;
         i + 7 <= Length;
         i++)
    {
        // 地址
        if (Buffer[i] !=
            Device_Address)
        {
            continue;
        }


        // Function 03
        if (Buffer[i + 1] !=
            0x03)
        {
            continue;
        }


        // 返回数据2字节
        if (Buffer[i + 2] !=
            0x02)
        {
            continue;
        }


        if (!Check_CRC(
                &Buffer[i],
                7))
        {
            Data.CRC_Error_Count++;

            continue;
        }


        // ====================================================
        // 距离：
        //
        // DATA_H在前
        // DATA_L在后
        //
        // 单位mm
        // ====================================================

        Data.Distance_mm =
            (
                static_cast<uint16_t>(
                    Buffer[i + 3]
                )
                << 8
            )
            |
            static_cast<uint16_t>(
                Buffer[i + 4]
            );


        Data.Distance_Valid =
            true;


        Data.Online =
            true;


        Data.Rx_Count++;


        Waiting_Response =
            false;


        Response_Timeout_Count =
            0;


        Offline_Count =
            0;


        return true;
    }


    return false;
}


// ============================================================
// Parse Write Response
//
// Function 06返回：
//
// 与发送帧完全相同
//
// 共8字节
// ============================================================

bool Class_TOF050F::Parse_Write_Response(
    const uint8_t *Buffer,
    uint16_t Length)
{
    if (Length < 8)
    {
        return false;
    }


    for (uint16_t i = 0;
         i + 8 <= Length;
         i++)
    {
        if (Buffer[i] !=
            Device_Address)
        {
            continue;
        }


        if (Buffer[i + 1] !=
            0x06)
        {
            continue;
        }


        if (!Check_CRC(
                &Buffer[i],
                8))
        {
            Data.CRC_Error_Count++;

            continue;
        }


        // ====================================================
        // 取写入的寄存器地址
        // ====================================================

        const uint16_t register_address =
            (
                static_cast<uint16_t>(
                    Buffer[i + 2]
                )
                << 8
            )
            |
            Buffer[i + 3];


        // ====================================================
        // 修改设备地址成功
        // ====================================================

        if (register_address ==
                TOF050F_REG_DEVICE_ADDRESS
            &&
            Pending_Address_Change)
        {
            Device_Address =
                Pending_New_Address;


            Pending_Address_Change =
                false;
        }


        Data.Online =
            true;


        Data.Rx_Count++;


        Waiting_Response =
            false;


        Response_Timeout_Count =
            0;


        Offline_Count =
            0;


        return true;
    }


    return false;
}


// ============================================================
// UART RX Callback
// ============================================================

void Class_TOF050F::UART_RxCpltCallback(
    uint8_t *Buffer,
    uint16_t Length)
{
    if (Buffer == nullptr ||
        Length == 0)
    {
        return;
    }


    // ========================================================
    // 优先解析03测距返回
    // ========================================================

    if (Parse_Read_Response(
            Buffer,
            Length))
    {
        return;
    }


    // ========================================================
    // 再解析06写寄存器返回
    // ========================================================

    if (Parse_Write_Response(
            Buffer,
            Length))
    {
        return;
    }


    // ========================================================
    // Modbus异常响应：
    //
    // Function = 原Function | 0x80
    //
    // 最短5字节
    // ========================================================

    for (uint16_t i = 0;
         i + 5 <= Length;
         i++)
    {
        if (Buffer[i] !=
            Device_Address)
        {
            continue;
        }


        if ((Buffer[i + 1] & 0x80) == 0)
        {
            continue;
        }


        if (Check_CRC(
                &Buffer[i],
                5))
        {
            Data.Frame_Error_Count++;


            Waiting_Response =
                false;


            Status =
                TOF050F_Status_ERROR;


            return;
        }
    }


    // ========================================================
    // 没有找到有效帧
    // ========================================================

    Data.Frame_Error_Count++;
}


// ============================================================
// 100ms callback
// ============================================================

void Class_TOF050F::
TIM_100ms_PeriodElapsedCallback()
{
    if (Status ==
        TOF050F_Status_DISABLE)
    {
        return;
    }


    // ========================================================
    // 等待回复超时处理
    //
    // 100ms调用一次
    //
    // 2次 = 200ms
    // ========================================================

    if (Waiting_Response)
    {
        Response_Timeout_Count++;


        if (Response_Timeout_Count >= 2)
        {
            Waiting_Response =
                false;


            Response_Timeout_Count =
                0;


            Data.Timeout_Count++;


            Offline_Count++;


            // 连续5次超时
            // 认为模块掉线

            if (Offline_Count >= 5)
            {
                Data.Online =
                    false;


                Data.Distance_Valid =
                    false;
            }
        }


        return;
    }


    // ========================================================
    // 自动主动测距
    // ========================================================

    if (Auto_Request_Enable)
    {
        Request_Distance();
    }
}