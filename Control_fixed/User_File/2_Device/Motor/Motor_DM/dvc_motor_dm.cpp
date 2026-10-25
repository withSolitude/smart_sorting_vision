/**
 * @file dvc_motor_dm.cpp
 * @author yssickjgd (1345578933@qq.com)
 * @brief 达妙电机配置与操作
 * @version 0.1
 * @date 2023-08-30 0.1 新增
 *
 * @copyright USTC-RoboWalker (c) 2023
 *
 */

/* Includes ------------------------------------------------------------------*/

#include "dvc_motor_dm.h"

#include <cmath>
#include <cstring>

/* Private macros ------------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private variables ---------------------------------------------------------*/

// 清除电机错误信息, 传统模式有效
uint8_t DM_Motor_CAN_Message_Clear_Error[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfb};
// 使能电机, 传统模式有效
uint8_t DM_Motor_CAN_Message_Enter[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc};
// 失能电机, 传统模式有效
uint8_t DM_Motor_CAN_Message_Exit[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd};
// 保存当前电机位置为零点, 传统模式有效
uint8_t DM_Motor_CAN_Message_Save_Zero[8] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe};

/* Private function declarations ---------------------------------------------*/

/* Function prototypes -------------------------------------------------------*/

/**
 * @brief 分配CAN发送缓冲区, 一拖四模式有效
 *
 * @param hcan CAN编号
 * @param __CAN_ID CAN ID
 * @param __DJI_Motor_Driver_Version 大疆驱动版本, 当且仅当当前被分配电机为6020, 且是电流驱动新版本时选2023, 否则都是default
 * @return uint8_t* 缓冲区指针
 */
uint8_t *allocate_tx_data(const FDCAN_HandleTypeDef *hcan, const Enum_Motor_DM_Motor_ID_1_To_4 &__CAN_Rx_ID_1_To_4)
{
    uint8_t *tmp_tx_data_ptr;
    if (hcan == &hfdcan1)
    {
        switch (__CAN_Rx_ID_1_To_4)
        {
        case (Motor_DM_ID_0x301):
        {
            tmp_tx_data_ptr = &(CAN1_0x3fe_Tx_Data[0]);

            break;
        }
        case (Motor_DM_ID_0x302):
        {
            tmp_tx_data_ptr = &(CAN1_0x3fe_Tx_Data[2]);

            break;
        }
        case (Motor_DM_ID_0x303):
        {
            tmp_tx_data_ptr = &(CAN1_0x3fe_Tx_Data[4]);

            break;
        }
        case (Motor_DM_ID_0x304):
        {
            tmp_tx_data_ptr = &(CAN1_0x3fe_Tx_Data[6]);

            break;
        }
        case (Motor_DM_ID_0x305):
        {
            tmp_tx_data_ptr = &(CAN1_0x4fe_Tx_Data[0]);

            break;
        }
        case (Motor_DM_ID_0x306):
        {
            tmp_tx_data_ptr = &(CAN1_0x4fe_Tx_Data[2]);

            break;
        }
        case (Motor_DM_ID_0x307):
        {
            tmp_tx_data_ptr = &(CAN1_0x4fe_Tx_Data[4]);

            break;
        }
        case (Motor_DM_ID_0x308):
        {
            tmp_tx_data_ptr = &(CAN1_0x4fe_Tx_Data[6]);

            break;
        }
        }
    }
    else if (hcan == &hfdcan2)
    {
        switch (__CAN_Rx_ID_1_To_4)
        {
        case (Motor_DM_ID_0x301):
        {
            tmp_tx_data_ptr = &(CAN2_0x3fe_Tx_Data[0]);

            break;
        }
        case (Motor_DM_ID_0x302):
        {
            tmp_tx_data_ptr = &(CAN2_0x3fe_Tx_Data[2]);

            break;
        }
        case (Motor_DM_ID_0x303):
        {
            tmp_tx_data_ptr = &(CAN2_0x3fe_Tx_Data[4]);

            break;
        }
        case (Motor_DM_ID_0x304):
        {
            tmp_tx_data_ptr = &(CAN2_0x3fe_Tx_Data[6]);

            break;
        }
        case (Motor_DM_ID_0x305):
        {
            tmp_tx_data_ptr = &(CAN2_0x4fe_Tx_Data[0]);

            break;
        }
        case (Motor_DM_ID_0x306):
        {
            tmp_tx_data_ptr = &(CAN2_0x4fe_Tx_Data[2]);

            break;
        }
        case (Motor_DM_ID_0x307):
        {
            tmp_tx_data_ptr = &(CAN2_0x4fe_Tx_Data[4]);

            break;
        }
        case (Motor_DM_ID_0x308):
        {
            tmp_tx_data_ptr = &(CAN2_0x4fe_Tx_Data[6]);

            break;
        }
        }
    }
    else if (hcan == &hfdcan3)
    {
        switch (__CAN_Rx_ID_1_To_4)
        {
        case (Motor_DM_ID_0x301):
        {
            tmp_tx_data_ptr = &(CAN3_0x3fe_Tx_Data[0]);

            break;
        }
        case (Motor_DM_ID_0x302):
        {
            tmp_tx_data_ptr = &(CAN3_0x3fe_Tx_Data[2]);

            break;
        }
        case (Motor_DM_ID_0x303):
        {
            tmp_tx_data_ptr = &(CAN3_0x3fe_Tx_Data[4]);

            break;
        }
        case (Motor_DM_ID_0x304):
        {
            tmp_tx_data_ptr = &(CAN3_0x3fe_Tx_Data[6]);

            break;
        }
        case (Motor_DM_ID_0x305):
        {
            tmp_tx_data_ptr = &(CAN3_0x4fe_Tx_Data[0]);

            break;
        }
        case (Motor_DM_ID_0x306):
        {
            tmp_tx_data_ptr = &(CAN3_0x4fe_Tx_Data[2]);

            break;
        }
        case (Motor_DM_ID_0x307):
        {
            tmp_tx_data_ptr = &(CAN3_0x4fe_Tx_Data[4]);

            break;
        }
        case (Motor_DM_ID_0x308):
        {
            tmp_tx_data_ptr = &(CAN3_0x4fe_Tx_Data[6]);

            break;
        }
        }
    }
    else
    {
        tmp_tx_data_ptr = nullptr;
    }
    return (tmp_tx_data_ptr);
}

/**
 * @brief 电机初始化
 *
 * @param hcan 绑定的CAN总线
 * @param __CAN_Rx_ID 收数据绑定的CAN ID, 与上位机驱动参数Master_ID保持一致, 传统模式有效
 * @param __CAN_Tx_ID 电机基础CAN ID, 与上位机CAN_ID保持一致；内部会自动生成控制帧ID
 * @param __Motor_DM_Control_Method 电机控制方式
 * @param __Angle_Max 最大位置, 与上位机控制幅值PMAX保持一致, 传统模式有效
 * @param __Omega_Max 最大速度, 与上位机控制幅值VMAX保持一致, 传统模式有效
 * @param __Torque_Max 最大扭矩, 与上位机控制幅值TMAX保持一致, 传统模式有效
 */
void Class_Motor_DM_Normal::Init(
    const FDCAN_HandleTypeDef *hcan,
    const uint8_t &__CAN_Rx_ID,
    const uint8_t &__CAN_Tx_ID,
    const Enum_Motor_DM_Control_Method &__Motor_DM_Control_Method,
    const float &__Angle_Max,
    const float &__Omega_Max,
    const float &__Torque_Max,
    const float &__Current_Max)
{
    // ========================================================
    // 1. 绑定CAN管理对象
    // ========================================================

    CAN_Manage_Object =
        nullptr;

    if (hcan == nullptr)
    {
        return;
    }

    if (hcan->Instance == FDCAN1)
    {
        CAN_Manage_Object =
            &CAN1_Manage_Object;
    }
    else if (hcan->Instance == FDCAN2)
    {
        CAN_Manage_Object =
            &CAN2_Manage_Object;
    }
    else if (hcan->Instance == FDCAN3)
    {
        CAN_Manage_Object =
            &CAN3_Manage_Object;
    }
    else
    {
        return;
    }


    // ========================================================
    // 2. 保存反馈帧ID
    //
    // 与达妙上位机Master ID一致。
    // ========================================================

    CAN_Rx_ID =
        __CAN_Rx_ID;


    // ========================================================
    // 3. 特殊命令帧ID
    //
    // 非常重要：
    //
    // 特殊命令：
    // FF FF FF FF FF FF FF FB  清错
    // FF FF FF FF FF FF FF FC  使能
    // FF FF FF FF FF FF FF FD  失能
    // FF FF FF FF FF FF FF FE  保存零点
    //
    // 使用“电机基础CAN ID”。
    //
    // 例如：
    // Motor ID = 0x01
    //
    // 特殊命令ID = 0x001
    //
    // 绝对不能发到0x101。
    // ========================================================

    CAN_Command_ID =
        __CAN_Tx_ID;


    // ========================================================
    // 4. 正常运动控制帧ID
    //
    // MIT：
    //      ID
    //
    // 位置速度：
    //      0x100 + ID
    //
    // 速度：
    //      0x200 + ID
    //
    // EMIT：
    //      0x300 + ID
    // ========================================================

    switch (__Motor_DM_Control_Method)
    {
        case Motor_DM_Control_Method_NORMAL_MIT:
        {
            CAN_Control_ID =
                __CAN_Tx_ID;

            break;
        }

        case Motor_DM_Control_Method_NORMAL_ANGLE_OMEGA:
        {
            CAN_Control_ID =
                static_cast<uint16_t>(
                    __CAN_Tx_ID
                    +
                    0x100U
                );

            break;
        }

        case Motor_DM_Control_Method_NORMAL_OMEGA:
        {
            CAN_Control_ID =
                static_cast<uint16_t>(
                    __CAN_Tx_ID
                    +
                    0x200U
                );

            break;
        }

        case Motor_DM_Control_Method_NORMAL_EMIT:
        {
            CAN_Control_ID =
                static_cast<uint16_t>(
                    __CAN_Tx_ID
                    +
                    0x300U
                );

            break;
        }

        default:
        {
            CAN_Control_ID =
                __CAN_Tx_ID;

            break;
        }
    }


    // ========================================================
    // 5. 控制参数
    // ========================================================

    Motor_DM_Control_Method =
        __Motor_DM_Control_Method;

    Angle_Max =
        __Angle_Max;

    Omega_Max =
        __Omega_Max;

    Torque_Max =
        __Torque_Max;

    Current_Max =
        __Current_Max;


    // ========================================================
    // 6. 运行状态全部显式清零
    //
    // 防止对象重新Init时残留旧状态。
    // ========================================================

    Flag =
        0U;

    Pre_Flag =
        0U;

    Feedback_Valid =
        false;
    Feedback_Update_Count = 0U;
    Feedback_Timestamp_ms = 0U;

    Output_Shaft_Angle =
        0.0f;

    Output_Shaft_Angle_Valid =
        false;

    Output_Shaft_Update_Count =
        0U;

    Parameter_Read_Pending =
        false;

    Pending_Read_Register_ID =
        0U;

    Motor_DM_Status =
        Motor_DM_Status_DISABLE;

    Rx_Data =
        {};

    Control_Angle =
        0.0f;

    Control_Omega =
        0.0f;

    Control_Torque =
        0.0f;

    Control_Current =
        0.0f;

    K_P =
        0.0f;

    K_D =
        0.0f;

    for (
        uint8_t i = 0U;
        i < 8U;
        ++i
    )
    {
        Tx_Data[i] =
            0U;
    }
}


/**
 * @brief CAN通信接收回调函数
 */
void Class_Motor_DM_Normal::CAN_RxCpltCallback()
{
    // ========================================================
    // 参数读取响应和普通状态反馈共用同一个Master ID。
    // 必须优先识别0x33寄存器响应；否则0x51(xout)返回的float
    // 会被当成MIT反馈帧解析，污染Now_Angle / Now_Omega。
    // ========================================================

    if (Parameter_Response_Process())
    {
        return;
    }


    // ========================================================
    // 只有真正属于本电机的反馈，
    // 才允许更新在线Flag。
    //
    // 原代码在Data_Process()之前Flag++，
    // 其他电机共用Master ID时可能造成误判在线。
    // ========================================================

    if (!Data_Process())
    {
        return;
    }


    Flag++;


    // 收到有效反馈以后立即认为电机在线，
    // 不必等待下一次100ms Alive周期。
    Motor_DM_Status =
        Motor_DM_Status_ENABLE;

    Feedback_Valid =
        true;
    Feedback_Timestamp_ms = HAL_GetTick();
    ++Feedback_Update_Count;
}


/**
 * @brief 发送清除错误信息
 *
 * 特殊命令帧必须使用CAN_Command_ID。
 */
void Class_Motor_DM_Normal::CAN_Send_Clear_Error()
{
    if (CAN_Manage_Object == nullptr)
    {
        return;
    }

    CAN_Transmit_Data(
        CAN_Manage_Object->CAN_Handler,
        CAN_Command_ID,
        DM_Motor_CAN_Message_Clear_Error,
        8
    );
}


/**
 * @brief 发送使能电机
 *
 * 特殊命令帧必须使用CAN_Command_ID。
 */
void Class_Motor_DM_Normal::CAN_Send_Enter()
{
    if (CAN_Manage_Object == nullptr)
    {
        return;
    }

    CAN_Transmit_Data(
        CAN_Manage_Object->CAN_Handler,
        CAN_Command_ID,
        DM_Motor_CAN_Message_Enter,
        8
    );
}


/**
 * @brief 发送失能电机
 *
 * 特殊命令帧必须使用CAN_Command_ID。
 */
void Class_Motor_DM_Normal::CAN_Send_Exit()
{
    if (CAN_Manage_Object == nullptr)
    {
        return;
    }

    CAN_Transmit_Data(
        CAN_Manage_Object->CAN_Handler,
        CAN_Command_ID,
        DM_Motor_CAN_Message_Exit,
        8
    );
}


/**
 * @brief 发送保存当前位置为零点
 *
 * 特殊命令帧必须使用CAN_Command_ID。
 */
void Class_Motor_DM_Normal::CAN_Send_Save_Zero()
{
    if (CAN_Manage_Object == nullptr)
    {
        return;
    }

    CAN_Transmit_Data(
        CAN_Manage_Object->CAN_Handler,
        CAN_Command_ID,
        DM_Motor_CAN_Message_Save_Zero,
        8
    );
}


/**
 * @brief 读取达妙驱动器寄存器。
 *
 * 请求格式（说明书）：
 * 0x7FF : CANID_L CANID_H 0x33 RID
 *
 * 这里只保存一个pending RID，避免普通MIT反馈被误判为寄存器响应。
 */
void Class_Motor_DM_Normal::CAN_Send_Read_Parameter(
    const uint8_t Register_ID)
{
    if (CAN_Manage_Object == nullptr)
    {
        return;
    }

    uint8_t tx[4] = {0U, 0U, 0U, 0U};

    tx[0] =
        static_cast<uint8_t>(CAN_Command_ID & 0xFFU);

    tx[1] =
        static_cast<uint8_t>((CAN_Command_ID >> 8U) & 0xFFU);

    tx[2] =
        0x33U;

    tx[3] =
        Register_ID;

    Pending_Read_Register_ID =
        Register_ID;

    Parameter_Read_Pending =
        true;

    CAN_Transmit_Data(
        CAN_Manage_Object->CAN_Handler,
        0x7FFU,
        tx,
        4U
    );
}


/**
 * @brief 尝试解析寄存器读取响应。
 *
 * 响应格式：
 * MST_ID : CANID_L CANID_H 0x33 RID DATA0 DATA1 DATA2 DATA3
 *
 * @return true 当前Rx_Buffer是本电机当前pending寄存器响应
 */
bool Class_Motor_DM_Normal::Parameter_Response_Process()
{
    if (
        CAN_Manage_Object == nullptr
    )
    {
        return false;
    }

    uint8_t *rx =
        reinterpret_cast<uint8_t *>(
            CAN_Manage_Object->Rx_Buffer
        );

    const uint16_t response_can_id =
        static_cast<uint16_t>(rx[0])
        |
        static_cast<uint16_t>(
            static_cast<uint16_t>(rx[1]) << 8U
        );

    if (
        response_can_id != CAN_Command_ID
        ||
        rx[2] != 0x33U
    )
    {
        return false;
    }

    // A late/duplicate register response must never be decoded as MIT
    // velocity/status feedback after the outstanding request has changed.
    if (!Parameter_Read_Pending || rx[3] != Pending_Read_Register_ID)
    {
        return true;
    }

    const uint8_t register_id =
        rx[3];

    const uint32_t raw_value =
        static_cast<uint32_t>(rx[4])
        |
        (static_cast<uint32_t>(rx[5]) << 8U)
        |
        (static_cast<uint32_t>(rx[6]) << 16U)
        |
        (static_cast<uint32_t>(rx[7]) << 24U);

    float value =
        0.0f;

    std::memcpy(
        &value,
        &raw_value,
        sizeof(value)
    );

    Parameter_Read_Pending =
        false;

    // 0x51 = xout：由输出轴编码器直接计算的输出轴位置。
    if (
        register_id == 0x51U
        &&
        std::isfinite(value)
    )
    {
        Output_Shaft_Angle =
            value;

        Output_Shaft_Angle_Valid =
            true;

        Output_Shaft_Update_Count++;
    }

    return true;
}


/**
 * @brief TIM定时器中断定期检测电机是否存活
 *
 * 建议100ms左右调用一次。
 *
 * 该函数负责：
 * 1. 在线检测
 * 2. 掉线/失能时低频重发使能
 * 3. 故障时低频清错
 *
 * 不再在2ms发送函数里狂发特殊命令。
 */
void Class_Motor_DM_Normal::TIM_Alive_PeriodElapsedCallback()
{
    // ========================================================
    // 1. 检查过去一个Alive周期有没有新反馈
    // ========================================================

    if (Flag == Pre_Flag)
    {
        Motor_DM_Status =
            Motor_DM_Status_DISABLE;
    }
    else
    {
        Motor_DM_Status =
            Motor_DM_Status_ENABLE;
    }

    Pre_Flag =
        Flag;


    // ========================================================
    // 2. 完全没有在线反馈：
    //
    // 每个Alive周期最多重发一次使能。
    // ========================================================

    if (
        Motor_DM_Status
        ==
        Motor_DM_Status_DISABLE
    )
    {
        CAN_Send_Enter();

        return;
    }


    // ========================================================
    // 3. 电机在线，但驱动状态不是Enable
    // ========================================================

    if (
        Rx_Data.Control_Status
        ==
        Motor_DM_Control_Status_DISABLE
    )
    {
        CAN_Send_Enter();

        return;
    }


    // ========================================================
    // 4. 电机在线但处于故障状态
    //
    // 先清错。
    // 下一周期如果回到DISABLE，再重新使能。
    // ========================================================

    if (
        Rx_Data.Control_Status
        !=
        Motor_DM_Control_Status_ENABLE
    )
    {
        CAN_Send_Clear_Error();
    }
}


/**
 * @brief TIM定时器周期发送运动控制帧
 *
 * 建议2ms左右调用一次。
 *
 * 该函数现在只负责“正常运动控制帧”，
 * 不再负责使能/清错。
 */
void Class_Motor_DM_Normal::TIM_Send_PeriodElapsedCallback()
{
    // ========================================================
    // 只有满足以下全部条件才允许发正常运动控制帧：
    //
    // 1. 已经收到过本电机有效反馈
    // 2. Alive判定在线
    // 3. 电机反馈状态为Enable
    // ========================================================

    if (!Feedback_Valid)
    {
        return;
    }

    if (
        Motor_DM_Status
        !=
        Motor_DM_Status_ENABLE
    )
    {
        return;
    }

    if (
        Rx_Data.Control_Status
        !=
        Motor_DM_Control_Status_ENABLE
    )
    {
        return;
    }


    // ========================================================
    // 控制量限幅
    // ========================================================

    Basic_Math_Constrain(
        &Control_Angle,
        -Angle_Max,
        Angle_Max
    );

    Basic_Math_Constrain(
        &Control_Omega,
        -Omega_Max,
        Omega_Max
    );

    Basic_Math_Constrain(
        &Control_Torque,
        -Torque_Max,
        Torque_Max
    );

    Basic_Math_Constrain(
        &Control_Current,
        -Current_Max,
        Current_Max
    );

    Basic_Math_Constrain(
        &K_P,
        0.0f,
        500.0f
    );

    Basic_Math_Constrain(
        &K_D,
        0.0f,
        5.0f
    );


    Output();
}


/**
 * @brief 数据处理过程
 *
 * @return true:
 *         当前Rx_Buffer属于本电机，数据已成功解析
 *
 * @return false:
 *         当前Rx_Buffer不是本电机反馈
 */
bool Class_Motor_DM_Normal::Data_Process()
{
    uint16_t tmp_encoder =
        0U;

    uint16_t tmp_omega =
        0U;

    uint16_t tmp_torque =
        0U;


    if (CAN_Manage_Object == nullptr)
    {
        return
            false;
    }


    Struct_Motor_DM_CAN_Rx_Data_Normal *tmp_buffer =
        reinterpret_cast<Struct_Motor_DM_CAN_Rx_Data_Normal *>(
            CAN_Manage_Object->Rx_Buffer
        );


    // ========================================================
    // D0低4位为电机CAN ID
    //
    // 这里必须与基础CAN ID比较，
    // 不再依赖“控制帧ID低4位”的写法。
    // ========================================================

    if (
        tmp_buffer->CAN_ID
        !=
        (
            CAN_Command_ID
            &
            0x0FU
        )
    )
    {
        return
            false;
    }


    // ========================================================
    // 处理大小端
    // ========================================================

    Basic_Math_Endian_Reverse_16(
        reinterpret_cast<void *>(
            &tmp_buffer->Angle_Reverse
        ),
        &tmp_encoder
    );


    tmp_omega =
        (
            static_cast<uint16_t>(
                tmp_buffer->Omega_11_4
            )
            <<
            4U
        )
        |
        (
            static_cast<uint16_t>(
                tmp_buffer->Omega_3_0_Torque_11_8
            )
            >>
            4U
        );


    tmp_torque =
        (
            (
                static_cast<uint16_t>(
                    tmp_buffer->Omega_3_0_Torque_11_8
                )
                &
                0x0FU
            )
            <<
            8U
        )
        |
        static_cast<uint16_t>(
            tmp_buffer->Torque_7_0
        );


    // ========================================================
    // 控制状态
    // ========================================================

    Rx_Data.Control_Status =
        static_cast<Enum_Motor_DM_Control_Status_Normal>(
            tmp_buffer->Control_Status_Enum
        );


    // ========================================================
    // 位置反馈
    //
    // 达妙传统模式反馈POS为16bit线性映射，
    // 映射范围由P_MAX确定。
    //
    // 这里直接还原驱动器反馈位置，
    // 不再自行做Total_Round跨圈累计。
    //
    // “圆形转盘走最短角度”的逻辑应该放在上层
    // sorting_storage中完成，而不是底层篡改反馈角度。
    // ========================================================

    Rx_Data.Now_Angle =
        Basic_Math_Int_To_Float(
            static_cast<int32_t>(
                tmp_encoder
            ),
            0,
            (1 << 16) - 1,
            -Angle_Max,
            Angle_Max
        );


    Rx_Data.Now_Omega =
        Basic_Math_Int_To_Float(
            static_cast<int32_t>(
                tmp_omega
            ),
            0,
            (1 << 12) - 1,
            -Omega_Max,
            Omega_Max
        );


    Rx_Data.Now_Torque =
        Basic_Math_Int_To_Float(
            static_cast<int32_t>(
                tmp_torque
            ),
            0,
            (1 << 12) - 1,
            -Torque_Max,
            Torque_Max
        );


    Rx_Data.Now_MOS_Temperature =
        tmp_buffer->MOS_Temperature
        +
        BASIC_MATH_CELSIUS_TO_KELVIN;


    Rx_Data.Now_Rotor_Temperature =
        tmp_buffer->Rotor_Temperature
        +
        BASIC_MATH_CELSIUS_TO_KELVIN;


    // ========================================================
    // 保留原始编码值，方便调试
    // ========================================================

    Rx_Data.Pre_Encoder =
        tmp_encoder;

    Rx_Data.Total_Round =
        0;

    Rx_Data.Total_Encoder =
        static_cast<int32_t>(
            tmp_encoder
        )
        -
        (
            (1 << 15)
            -
            1
        );


    return
        true;
}


/**
 * @brief 电机数据输出到CAN总线
 *
 * 这里只能发送“正常运动控制帧”。
 *
 * 特殊命令帧由CAN_Send_xxx()使用CAN_Command_ID发送。
 */
void Class_Motor_DM_Normal::Output()
{
    if (CAN_Manage_Object == nullptr)
    {
        return;
    }


    switch (Motor_DM_Control_Method)
    {
        // ====================================================
        // MIT
        //
        // 控制帧ID = 基础ID
        // ====================================================

        case Motor_DM_Control_Method_NORMAL_MIT:
        {
            Struct_Motor_DM_CAN_Tx_Data_Normal_MIT *tmp_buffer =
                reinterpret_cast<Struct_Motor_DM_CAN_Tx_Data_Normal_MIT *>(
                    Tx_Data
                );


            uint16_t tmp_angle =
                static_cast<uint16_t>(
                    Basic_Math_Float_To_Int(
                        Control_Angle,
                        -Angle_Max,
                        Angle_Max,
                        0,
                        (1 << 16) - 1
                    )
                );


            uint16_t tmp_omega =
                static_cast<uint16_t>(
                    Basic_Math_Float_To_Int(
                        Control_Omega,
                        -Omega_Max,
                        Omega_Max,
                        0,
                        (1 << 12) - 1
                    )
                );


            uint16_t tmp_torque =
                static_cast<uint16_t>(
                    Basic_Math_Float_To_Int(
                        Control_Torque,
                        -Torque_Max,
                        Torque_Max,
                        0,
                        (1 << 12) - 1
                    )
                );


            uint16_t tmp_k_p =
                static_cast<uint16_t>(
                    Basic_Math_Float_To_Int(
                        K_P,
                        0.0f,
                        500.0f,
                        0,
                        (1 << 12) - 1
                    )
                );


            uint16_t tmp_k_d =
                static_cast<uint16_t>(
                    Basic_Math_Float_To_Int(
                        K_D,
                        0.0f,
                        5.0f,
                        0,
                        (1 << 12) - 1
                    )
                );


            tmp_buffer->Control_Angle_Reverse =
                Basic_Math_Endian_Reverse_16(
                    &tmp_angle,
                    nullptr
                );


            tmp_buffer->Control_Omega_11_4 =
                static_cast<uint8_t>(
                    tmp_omega
                    >>
                    4U
                );


            tmp_buffer->Control_Omega_3_0_K_P_11_8 =
                static_cast<uint8_t>(
                    (
                        (
                            tmp_omega
                            &
                            0x0FU
                        )
                        <<
                        4U
                    )
                    |
                    (
                        tmp_k_p
                        >>
                        8U
                    )
                );


            tmp_buffer->K_P_7_0 =
                static_cast<uint8_t>(
                    tmp_k_p
                    &
                    0xFFU
                );


            tmp_buffer->K_D_11_4 =
                static_cast<uint8_t>(
                    tmp_k_d
                    >>
                    4U
                );


            tmp_buffer->K_D_3_0_Control_Torque_11_8 =
                static_cast<uint8_t>(
                    (
                        (
                            tmp_k_d
                            &
                            0x0FU
                        )
                        <<
                        4U
                    )
                    |
                    (
                        tmp_torque
                        >>
                        8U
                    )
                );


            tmp_buffer->Control_Torque_7_0 =
                static_cast<uint8_t>(
                    tmp_torque
                    &
                    0xFFU
                );


            CAN_Transmit_Data(
                CAN_Manage_Object->CAN_Handler,
                CAN_Control_ID,
                Tx_Data,
                8
            );


            break;
        }


        // ====================================================
        // 位置速度模式
        //
        // 控制帧ID = 0x100 + ID
        //
        // D0~D3 = p_des(float, little endian)
        // D4~D7 = v_des(float, little endian)
        // ====================================================

        case Motor_DM_Control_Method_NORMAL_ANGLE_OMEGA:
        {
            Struct_Motor_DM_CAN_Tx_Data_Normal_Angle_Omega *tmp_buffer =
                reinterpret_cast<Struct_Motor_DM_CAN_Tx_Data_Normal_Angle_Omega *>(
                    Tx_Data
                );


            tmp_buffer->Control_Angle =
                Control_Angle;


            tmp_buffer->Control_Omega =
                Control_Omega;


            CAN_Transmit_Data(
                CAN_Manage_Object->CAN_Handler,
                CAN_Control_ID,
                Tx_Data,
                8
            );


            break;
        }


        // ====================================================
        // 速度模式
        //
        // 控制帧ID = 0x200 + ID
        // ====================================================

        case Motor_DM_Control_Method_NORMAL_OMEGA:
        {
            Struct_Motor_DM_CAN_Tx_Data_Normal_Omega *tmp_buffer =
                reinterpret_cast<Struct_Motor_DM_CAN_Tx_Data_Normal_Omega *>(
                    Tx_Data
                );


            tmp_buffer->Control_Omega =
                Control_Omega;


            CAN_Transmit_Data(
                CAN_Manage_Object->CAN_Handler,
                CAN_Control_ID,
                Tx_Data,
                4
            );


            break;
        }


        // ====================================================
        // EMIT
        // ====================================================

        case Motor_DM_Control_Method_NORMAL_EMIT:
        {
            Struct_Motor_DM_CAN_Tx_Data_Normal_EMIT *tmp_buffer =
                reinterpret_cast<Struct_Motor_DM_CAN_Tx_Data_Normal_EMIT *>(
                    Tx_Data
                );


            tmp_buffer->Control_Angle =
                Control_Angle;


            tmp_buffer->Control_Omega =
                static_cast<uint16_t>(
                    Control_Omega
                    *
                    100.0f
                );


            tmp_buffer->Control_Current =
                static_cast<uint16_t>(
                    Control_Current
                    /
                    Current_Max
                    *
                    10000.0f
                );


            CAN_Transmit_Data(
                CAN_Manage_Object->CAN_Handler,
                CAN_Control_ID,
                Tx_Data,
                8
            );


            break;
        }


        default:
        {
            break;
        }
    }
}


/**
 * @brief 电机初始化
 *
 * @param hcan 绑定的CAN总线
 * @param __CAN_Rx_ID 绑定的CAN ID
 * @param __Motor_DM_Control_Method 电机控制方式, 默认角度
 * @param __Encoder_Offset 编码器偏移, 默认0
 * @param __Current_Max 最大电流
 */
void Class_Motor_DM_1_To_4::Init(const FDCAN_HandleTypeDef *hcan, const Enum_Motor_DM_Motor_ID_1_To_4 &__CAN_Rx_ID, const Enum_Motor_DM_Control_Method &__Motor_DM_Control_Method, const int32_t &__Encoder_Offset, const float &__Current_Max)
{
    if (hcan->Instance == FDCAN1)
    {
        CAN_Manage_Object = &CAN1_Manage_Object;
    }
    else if (hcan->Instance == FDCAN2)
    {
        CAN_Manage_Object = &CAN2_Manage_Object;
    }
    else if (hcan->Instance == FDCAN3)
    {
        CAN_Manage_Object = &CAN3_Manage_Object;
    }
    CAN_Rx_ID = __CAN_Rx_ID;
    Motor_DM_Control_Method = __Motor_DM_Control_Method;
    Encoder_Offset = __Encoder_Offset;
    Current_Max = __Current_Max;
    Tx_Data = allocate_tx_data(hcan, __CAN_Rx_ID);
}

/**
 * @brief CAN通信接收回调函数
 *
 */
void Class_Motor_DM_1_To_4::CAN_RxCpltCallback()
{
    // 滑动窗口, 判断电机是否在线
    Flag += 1;

    Data_Process();
}

/**
 * @brief TIM定时器中断定期检测电机是否存活
 *
 */
void Class_Motor_DM_1_To_4::TIM_100ms_Alive_PeriodElapsedCallback()
{
    // 判断该时间段内是否接收过电机数据
    if (Flag == Pre_Flag)
    {
        // 电机断开连接
        Motor_DM_Status = Motor_DM_Status_DISABLE;
        PID_Angle.Set_Integral_Error(0.0f);
        PID_Omega.Set_Integral_Error(0.0f);
    }
    else
    {
        // 电机保持连接
        Motor_DM_Status = Motor_DM_Status_ENABLE;
    }
    Pre_Flag = Flag;
}

/**
 * @brief TIM定时器中断计算回调函数, 计算周期取决于电机反馈周期
 *
 */
void Class_Motor_DM_1_To_4::TIM_1ms_Calculate_PeriodElapsedCallback()
{
    PID_Calculate();

    float tmp_value = Target_Current + Feedforward_Current;
    Basic_Math_Constrain(&tmp_value, -Current_Max, Current_Max);
    Out = tmp_value * Current_To_Out;

    Output();

    Feedforward_Current = 0.0f;
    Feedforward_Omega = 0.0f;
}

/**
 * @brief 数据处理过程
 *
 */
void Class_Motor_DM_1_To_4::Data_Process()
{
    // 数据处理过程
    int16_t delta_encoder;
    uint16_t tmp_encoder;
    int16_t tmp_omega, tmp_current;
    Struct_Motor_DM_CAN_Rx_Data_1_To_4 *tmp_buffer = (Struct_Motor_DM_CAN_Rx_Data_1_To_4 *) CAN_Manage_Object->Rx_Buffer;

    // 处理大小端
    Basic_Math_Endian_Reverse_16((void *) &tmp_buffer->Encoder_Reverse, (void *) &tmp_encoder);
    Basic_Math_Endian_Reverse_16((void *) &tmp_buffer->Omega_Reverse, (void *) &tmp_omega);
    Basic_Math_Endian_Reverse_16((void *) &tmp_buffer->Current_Reverse, (void *) &tmp_current);

    // 计算圈数与总编码器值
    delta_encoder = tmp_encoder - Rx_Data.Pre_Encoder;
    if (delta_encoder < -Encoder_Num_Per_Round / 2)
    {
        // 正方向转过了一圈
        Rx_Data.Total_Round++;
    }
    else if (delta_encoder > Encoder_Num_Per_Round / 2)
    {
        // 反方向转过了一圈
        Rx_Data.Total_Round--;
    }
    Rx_Data.Total_Encoder = Rx_Data.Total_Round * Encoder_Num_Per_Round + tmp_encoder + Encoder_Offset;

    // 计算电机本身信息
    Rx_Data.Now_Angle = (float) Rx_Data.Total_Encoder / (float) Encoder_Num_Per_Round * 2.0f * PI;
    Rx_Data.Now_Omega = tmp_omega / 100.0f * BASIC_MATH_RPM_TO_RADPS;
    Rx_Data.Now_Current = tmp_current / 1000.0f;
    Rx_Data.Now_MOS_Temperature = tmp_buffer->MOS_Temperature + BASIC_MATH_CELSIUS_TO_KELVIN;
    Rx_Data.Now_Rotor_Temperature = tmp_buffer->Rotor_Temperature + BASIC_MATH_CELSIUS_TO_KELVIN;

    // 存储预备信息
    Rx_Data.Pre_Encoder = tmp_encoder;
}

/**
 * @brief 计算PID
 *
 */
void Class_Motor_DM_1_To_4::PID_Calculate()
{
    switch (Motor_DM_Control_Method)
    {
    case (Motor_DM_Control_Method_1_TO_4_CURRENT):
    {
        break;
    }
    case (Motor_DM_Control_Method_1_TO_4_OMEGA):
    {
        PID_Omega.Set_Target(Target_Omega + Feedforward_Omega);
        PID_Omega.Set_Now(Rx_Data.Now_Omega);
        PID_Omega.TIM_Calculate_PeriodElapsedCallback();

        Target_Current = PID_Omega.Get_Out();

        break;
    }
    case (Motor_DM_Control_Method_1_TO_4_ANGLE):
    {
        PID_Angle.Set_Target(Target_Angle);
        PID_Angle.Set_Now(Rx_Data.Now_Angle);
        PID_Angle.TIM_Calculate_PeriodElapsedCallback();

        Target_Omega = PID_Angle.Get_Out();

        PID_Omega.Set_Target(Target_Omega + Feedforward_Omega);
        PID_Omega.Set_Now(Rx_Data.Now_Omega);
        PID_Omega.TIM_Calculate_PeriodElapsedCallback();

        Target_Current = PID_Omega.Get_Out();

        break;
    }
    default:
    {
        Target_Current = 0.0f;

        break;
    }
    }
}

/**
 * @brief 电机数据输出到CAN总线发送缓冲区
 *
 */
void Class_Motor_DM_1_To_4::Output()
{
    *(int16_t *) Tx_Data = (int16_t)(Out);
}

/************************ COPYRIGHT(C) USTC-ROBOWALKER **************************/
