//
// Created by Lenovo on 2026/8/28.
//

#include "3_Robot/holding_jaw.h"
#include "3_Robot/claw_preopen_profile.h"
#include <cmath>


// ============================================================
// Init
// ============================================================

void Class_Holding_Jaw::Init()
{
    Grap_Error = Enum_Grap_Error_None;
    Grap_State_Start_Tick = HAL_GetTick();
    Fault_Lift_Stop_Pending = false;
    Fault_Yaw_Stop_Pending = false;
    Calibration_Yaw_Override = false;

    // ========================================================
    // 舵机初始化
    // ========================================================

    Yaw_Servo_Motor.Init(
        &huart3,
        0,
        Servo_Motor_Work_Mode_270_CW
    );


    Lift_Servo_Motor.Init(
        &huart3,
        1,
        Servo_Motor_Work_Mode_270_CW
    );


    Claw_Servo_Motor.Init(
        &huart3,
        2,
        Servo_Motor_Work_Mode_270_CW
    );


    // CoreXY A/B 电机使用 CAN；在人工确认原点前，位置保持无效。
    // 机械尺度、电机方向和 CAN 配置统一位于 xy_motion_config.h。
    XY_Motion.Init();

    // ========================================================
    // 初始状态
    // ========================================================

    Grap_Move_Status =
        Enum_Grap_Move_Disable;

    // 复位后每次都重新执行一次“反馈有效后自动回 Disable”。
    Power_On_Disable_Pending = true;


    Grap_Move_Count =
        0;


    Grap_Output_Flag =
        false;


    Grap_Output_Step =
        0;


    // ========================================================
    // 初始舵机目标
    // ========================================================

    Yaw_Servo_Motor_Target_Angle =
        Yaw_Servo_Motor_Angle_Disable;


    Lift_Servo_Motor_Target_Angle =
        Lift_Servo_Motor_Angle_Disable;


    Claw_Servo_Motor_Target_Angle =
        Claw_Servo_Motor_Angle_Disable;


    // ========================================================
    // TOF下降变量初始化
    // ========================================================

    Lift_Descend_Target_Angle =
        Lift_Servo_Motor_Angle_Lift;


    Grap_TOF_Distance_mm =
        0;


    Grap_TOF_Reached =
        false;


    // Lift下降到软件机械下限，并且最终下限命令已经成功发给舵机。
    // 该标志用于“TOF未触发但已经到底”的兜底夹取。
    Grap_Lift_Limit_Commanded =
        false;


    Grap_Failed =
        false;


    Grap_TOF_Stable_Count =
        0;


    Grap_TOF_Last_Rx_Count =
        0;


    Grap_TOF_No_Update_Count_ms =
        0;


    Lift_Control_Send_Count =
        0;
}


// ============================================================
// 1ms Callback
//
// 只负责第一次抓物体时的TOF闭环下降
// ============================================================

void Class_Holding_Jaw::
TIM_1ms_PeriodElapsedCallback()
{
    XY_Motion.TIM_1ms_PeriodElapsedCallback();

    if (Grap_Move_Status != Enum_Grap_Move_Disable &&
        Grap_Move_Status != Enum_Grap_Move_Fault &&
        (XY_Motion.Get_Fault() || !XY_Motion.Get_Position_Valid()))
    {
        Abort_XY_Fault();
        return;
    }

    if (Grap_Move_Status == Enum_Grap_Move_Fault)
    {
        Service_Fault_Servo_Stop();
        return;
    }

    // ========================================================
    // 只有Descend状态运行
    //
    // 识别托盘下降不走这里。
    // ========================================================

    if (Grap_Move_Status !=
        Enum_Grap_Move_Descend)
    {
        return;
    }


    // ========================================================
    // TOF有效性也必须参与“最后有效数据”超时。
    //
    // 旧逻辑在valid=false时直接return，计时器永远不增长，可能永久卡在DESCEND。
    // 现在无效数据立即冻结继续下降，并累计无效时长；达到阈值后安全Return。
    // ========================================================

    if (!TOF050F.Get_Distance_Valid())
    {
        Grap_TOF_Stable_Count = 0;
        Grap_TOF_Reached = false;

        if (Grap_TOF_No_Update_Count_ms < 65535)
        {
            Grap_TOF_No_Update_Count_ms++;
        }

        if (Grap_TOF_No_Update_Count_ms >= TOF_Data_Timeout_ms)
        {
            Grap_Error = Enum_Grap_Error_TOF_Timeout;
            Grap_Failed = true;
            Set_Grap_Move_Status(Enum_Grap_Move_Return);
        }

        // 无有效距离时不再更新Lift下降目标。
        return;
    }


    // ========================================================
    // 检查有没有一帧新的TOF数据
    // ========================================================

    const uint32_t current_rx_count =
        TOF050F.Get_Data().Rx_Count;


    if (current_rx_count !=
        Grap_TOF_Last_Rx_Count)
    {
        // ----------------------------------------------------
        // 收到新的距离数据
        // ----------------------------------------------------

        Grap_TOF_Last_Rx_Count =
            current_rx_count;


        Grap_TOF_No_Update_Count_ms =
            0;


        Grap_TOF_Distance_mm =
            TOF050F.Get_Distance_mm();


        // ----------------------------------------------------
        // 新数据已经达到抓取距离
        // ----------------------------------------------------

        if (Grap_TOF_Distance_mm <=
            Grap_TOF_Trigger_Distance_mm)
        {
            if (Grap_TOF_Stable_Count <
                255)
            {
                Grap_TOF_Stable_Count++;
            }


            // ------------------------------------------------
            // 连续两帧满足
            //
            // 才认为真正达到抓取位置
            // ------------------------------------------------

            if (Grap_TOF_Stable_Count >=
                Grap_TOF_Stable_Required)
            {
                Grap_TOF_Reached =
                    true;
            }
        }
        else
        {
            Grap_TOF_Stable_Count =
                0;


            Grap_TOF_Reached =
                false;
        }
    }
    else
    {
        // ====================================================
        // 没有新TOF数据
        // ====================================================

        if (Grap_TOF_No_Update_Count_ms <
            65535)
        {
            Grap_TOF_No_Update_Count_ms++;
        }
    }


    // ========================================================
    // 已经达到物体
    //
    // 立即停止更新Lift目标。
    // ========================================================

    if (Grap_TOF_Reached || Grap_Lift_Limit_Commanded)
    {
        return;
    }


    // ========================================================
    // TOF数据太久没有更新
    //
    // 为安全起见停止继续下降。
    // ========================================================

    if (Grap_TOF_No_Update_Count_ms >=
        TOF_Data_Timeout_ms)
    {
        Grap_Error = Enum_Grap_Error_TOF_Timeout;
        Grap_Failed = true;
        Set_Grap_Move_Status(Enum_Grap_Move_Return);
        return;
    }


    // ========================================================
    // Lift按照固定角速度持续下降
    //
    // 当前：
    //
    // 10° / s
    //
    // 1ms一次：
    //
    // delta =
    // 10°/s × 0.001s
    //
    // = 0.01°
    // ========================================================

    Lift_Descend_Target_Angle +=
        Lift_Descend_Speed *
        0.001f;


    // ========================================================
    // 机械下降极限
    //
    // 重要：
    // TOF距离没有达到阈值，并不代表最低位置一定没有物体。
    // 以前这里直接标记失败并Return，所以会出现“下降到底却不夹，直接回零”。
    //
    // 现在改为：
    // 1) 目标最多到Lift_Descend_Max_Angle；
    // 2) 继续把这个最终下限目标可靠发给Lift；
    // 3) 下限命令发送成功后，由100ms状态机进入Clip，强制收紧夹爪。
    //
    // 注意：这里只对“有效TOF一直大于阈值、但Lift已经到底”做兜底。
    // TOF失效/超时仍然按故障处理，不盲目继续下降。
    // ========================================================

    if (Lift_Descend_Target_Angle >=
        Lift_Descend_Max_Angle)
    {
        Lift_Descend_Target_Angle =
            Lift_Descend_Max_Angle;
    }


    // ========================================================
    // 1ms内部计算
    //
    // 但不是每1ms发UART
    //
    // 20ms发一次。
    // ========================================================

    Lift_Control_Send_Count++;


    if (Lift_Control_Send_Count <
        Lift_Control_Send_Period_ms)
    {
        return;
    }


    Lift_Control_Send_Count =
        0;


    // ========================================================
    // USART3必须空闲
    // ========================================================

    if (huart3.gState !=
        HAL_UART_STATE_READY)
    {
        return;
    }


    // ========================================================
    // 更新Lift目标
    // ========================================================

    Lift_Servo_Motor_Target_Angle =
        Lift_Descend_Target_Angle;


    // ========================================================
    // 给Lift发送新的动态目标
    //
    // 20ms更新一次
    //
    // 每次执行时间50ms
    //
    // 形成连续平滑下降。
    // ========================================================

    const uint8_t lift_tx_status =
        Lift_Servo_Motor.Control_Angle(
            Lift_Servo_Motor_Target_Angle,

            Lift_Control_Move_Time_ms
        );


    // ========================================================
    // 如果已经下降到软件机械下限，并且这个最终下限命令已经成功发出，
    // 就允许100ms状态机进入Clip。
    //
    // 这里不直接切状态，是为了至少保证最后一个Lift下限命令已经进入发送层，
    // 再让夹爪闭合，避免“最后几毫米还没下完就先夹”。
    // ========================================================

    if (lift_tx_status == HAL_OK &&
        Lift_Descend_Target_Angle >=
            (Lift_Descend_Max_Angle - 0.0001f) && !Grap_Lift_Limit_Commanded)
    {
        Grap_Lift_Limit_Command_Tick = HAL_GetTick();
        Grap_Lift_Limit_Commanded =
            true;
    }
}


// ============================================================
// 100ms抓取任务
// ============================================================

void Class_Holding_Jaw::
TIM_100ms_PeriodElapsedCallback()
{
    // ========================================================
    // 先执行当前状态输出
    // ========================================================

    Outputs();


    // ========================================================
    // 再判断状态是否完成
    // ========================================================

    Grap_Move();

    // ========================================================
    // 上电自动回到 Disable XY 坐标
    //
    // XY_Motion.Init() 后必须先收到两台 EMM 的有效反馈，
    // 上电软件原点才会建立。此时不能在 Init() 里直接调用
    // Move_To_mm()，否则会因为 Position_Valid=false 被拒绝。
    // 等舵机待机命令稳定、XY反馈有效后再走统一 Return 状态机，
    // 这样与STOP/故障回位使用完全相同的角度和速度参数。
    // ========================================================
    if (Power_On_Disable_Pending &&
        Grap_Move_Status == Enum_Grap_Move_Disable)
    {
        if (Get_Disable_Ready())
        {
            // 上电时已经在 Disable，清掉一次性请求，不再重复下发移动。
            Power_On_Disable_Pending = false;
        }
        else if (XY_Motion.Get_Position_Valid() &&
                 !XY_Motion.Get_Fault() &&
                 Motion_Configuration_Valid() &&
                 Get_Stop_Ready())
        {
            // Request_Return_To_Disable() 内部会把状态切换到 Return，
            // 下一次100ms回调由 Outputs()发送 XY 的 Disable 坐标。
            const HAL_StatusTypeDef status = Request_Return_To_Disable();
            if (status == HAL_OK)
            {
                Power_On_Disable_Pending = false;
            }
            // HAL_BUSY 表示反馈/舵机仍在稳定，下一周期继续尝试；
            // HAL_ERROR 也保留请求，避免一次时序抖动导致永远不回位。
        }
    }
}


// ============================================================
// Grap Move
// ============================================================

void Class_Holding_Jaw::Grap_Move()
{
    const bool active_phase = Grap_Move_Status != Enum_Grap_Move_Disable &&
        Grap_Move_Status != Enum_Grap_Move_Finish && Grap_Move_Status != Enum_Grap_Move_Fault;
    if (active_phase &&
        (uint32_t)(HAL_GetTick() - Grap_State_Start_Tick) >= XY_State_Timeout_ms)
    {
        Abort_Grap_Fault(Enum_Grap_Error_Actuator_Timeout);
        return;
    }

    // UART DMA 提交成功不等于 500 ms 舵机动作已经完成。
    if (Grap_Output_Flag && !Get_Outputs_Settled()) return;

    switch (Grap_Move_Status)
    {
        // ====================================================
        // DISABLE
        // ====================================================

        case Enum_Grap_Move_Disable:
        {
            Grap_Move_Count =
                0;


            break;
        }


        // ====================================================
        // 1.
        //
        // X/Y/Yaw移动到物体
        // ====================================================

        case Enum_Grap_Move_X_Y_Start:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (XY_Move_Finished() &&
                Grap_Move_Count >= 5)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Pre_Clip
                );
            }


            break;
        }


        // ====================================================
        // 2.
        //
        // Claw预收缩
        // ====================================================

        case Enum_Grap_Move_Pre_Clip:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (Grap_Move_Count >= 5)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Descend
                );
            }


            break;
        }


        // ====================================================
        // 3.
        //
        // TOF闭环下降
        //
        // 这里不再按照时间切换。
        //
        // 1ms控制器检测：
        //
        // TOF距离连续两帧 <= 阈值
        //
        // 才将：
        //
        // Grap_TOF_Reached = true
        // ====================================================

        case Enum_Grap_Move_Descend:
        {
            // 正常路径：TOF连续达到阈值。
            // 兜底路径：TOF仍未到阈值，但Lift已经下降到限定角度，
            // 且最终下限命令已经成功发出。两种情况都必须进入正式夹紧。
            if (Grap_TOF_Reached ||
                (Grap_Lift_Limit_Commanded &&
                 (uint32_t)(HAL_GetTick() - Grap_Lift_Limit_Command_Tick) >= Lift_Control_Move_Time_ms))
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Clip
                );
            }


            break;
        }


        // ====================================================
        // 4.
        //
        // 正式夹紧
        // ====================================================

        case Enum_Grap_Move_Clip:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (Grap_Move_Count >= 10)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Lift
                );
            }


            break;
        }


        // ====================================================
        // 5.
        //
        // 抓住物体以后Lift抬升
        // ====================================================

        case Enum_Grap_Move_Lift:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (Grap_Move_Count >= 10)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_X_Y_Tray
                );
            }


            break;
        }


        // ====================================================
        // 6.
        //
        // X/Y/Yaw移动到识别托盘
        // ====================================================

        case Enum_Grap_Move_X_Y_Tray:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (XY_Move_Finished() &&
                Grap_Move_Count >= 5)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Tray_Descend
                );
            }


            break;
        }


        // ====================================================
        // 7.
        //
        // 识别托盘固定位置下降
        //
        // 这个地方不使用TOF。
        // ====================================================

        case Enum_Grap_Move_Tray_Descend:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (Grap_Move_Count >= 10)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Release
                );
            }


            break;
        }


        // ====================================================
        // 8.
        //
        // Claw释放
        // ====================================================

        case Enum_Grap_Move_Release:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (Grap_Move_Count >= 10)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Tray_Lift
                );
            }


            break;
        }


        // ====================================================
        // 9.
        //
        // 放下以后重新抬升
        // ====================================================

        case Enum_Grap_Move_Tray_Lift:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (Grap_Move_Count >= 10)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Recognition_Position
                );
            }


            break;
        }


        // ====================================================
        // 10.
        //
        // 移动到识别姿态。
        //
        // 正常流程到这里以后先保持识别姿态，供侧面相机识别。
        // 收到本件 SORT_RESULT 后，由 ita_robot.cpp 请求回到 Disable；
        // 下一件物体只有在 Disable 完成后才允许开始抓取。
        // ====================================================

        case Enum_Grap_Move_Recognition_Position:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (XY_Move_Finished() &&
                Grap_Move_Count >= 10)
            {
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Finish
                );
            }


            break;
        }


        // ====================================================
        // 安全/STOP专用：整机回Disable
        // ====================================================

        case Enum_Grap_Move_Return:
        {
            if (!Grap_Output_Flag)
            {
                Grap_Move_Count =
                    0;


                break;
            }


            Grap_Move_Count = static_cast<uint16_t>(
                (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) / 100U + 1U);


            if (XY_Move_Finished() &&
                Grap_Move_Count >= 10)
            {
                Calibration_Yaw_Override = false;
                Set_Grap_Move_Status(
                    Enum_Grap_Move_Disable
                );
            }


            break;
        }


        // ====================================================
        // FINISH
        //
        // 正常流程在识别姿态完成后停留在这里，等待侧面识别。
        // 收到本件 SORT_RESULT 后会请求 Return -> Disable。
        // 禁止直接从 Finish 启动下一件抓取，避免遮挡顶部相机。
        // ====================================================

        case Enum_Grap_Move_Finish:
        case Enum_Grap_Move_Fault:
        {
            break;
        }


        // ====================================================
        // 异常
        // ====================================================

        default:
        {
            Abort_XY_Fault();


            break;
        }
    }
}


// ============================================================
// 输出当前夹爪状态指令
// ============================================================

void Class_Holding_Jaw::Outputs()
{
    uint8_t tx_status =
        HAL_BUSY;


    HAL_StatusTypeDef xy_status = HAL_ERROR;


    switch (Grap_Move_Status)
    {
        // ====================================================
        // DISABLE
        //
        // 不是失能
        //
        // Yaw -> Disable
        // Lift -> Disable
        // Claw -> Disable
        // ====================================================

        case Enum_Grap_Move_Disable:
        {
            // ------------------------------------------------
            // Step 0
            //
            // Yaw
            // ------------------------------------------------

            if (Grap_Output_Step == 0)
            {
                Yaw_Servo_Motor_Target_Angle =
                    Yaw_Servo_Motor_Angle_Disable;


                tx_status =
                    Yaw_Servo_Motor.Control_Angle(
                        Yaw_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        1;
                }


                break;
            }


            // ------------------------------------------------
            // Step 1
            //
            // Lift
            // ------------------------------------------------

            if (Grap_Output_Step == 1)
            {
                Lift_Servo_Motor_Target_Angle =
                    Lift_Servo_Motor_Angle_Disable;


                tx_status =
                    Lift_Servo_Motor.Control_Angle(
                        Lift_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        2;
                }


                break;
            }


            // ------------------------------------------------
            // Step 2
            //
            // Claw
            // ------------------------------------------------

            if (Grap_Output_Step == 2)
            {
                Claw_Servo_Motor_Target_Angle =
                    Claw_Servo_Motor_Angle_Disable;


                tx_status =
                    Claw_Servo_Motor.Control_Angle(
                        Claw_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        3;


                    Complete_Outputs();
                }


                break;
            }


            break;
        }


        // ====================================================
        // X/Y/Yaw去目标物体
        // ====================================================

        case Enum_Grap_Move_X_Y_Start:
        {
            // ------------------------------------------------
            // Step 0
            //
            // X/Y同时开始
            // ------------------------------------------------

            if (Grap_Output_Step == 0)
            {
                xy_status = XY_Motion.Move_To_mm(
                    Grap_Target_X_mm, Grap_Target_Y_mm, X_Move_Speed_mm_s, Y_Move_Speed_mm_s);

                if (xy_status == HAL_OK)
                {
                    Grap_Output_Step =
                        1;
                }
                else
                {
                    if (xy_status != HAL_BUSY)
                    {
                        Abort_XY_Fault();
                    }
                    break;
                }
            }


            // ------------------------------------------------
            // Step 1
            //
            // Yaw
            // ------------------------------------------------

            if (Grap_Output_Step == 1)
            {
                Yaw_Servo_Motor_Target_Angle =
                    Grap_Target_Yaw_Angle;


                tx_status =
                    Yaw_Servo_Motor.Control_Angle(
                        Yaw_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        2;


                    Complete_Outputs();
                }


                break;
            }


            break;
        }


        // ====================================================
        // Claw预收缩
        // ====================================================

        case Enum_Grap_Move_Pre_Clip:
        {
            if (Grap_Output_Flag)
            {
                break;
            }


            Claw_Servo_Motor_Target_Angle =
                Grap_Target_Pre_Clip_Angle;


            tx_status =
                Claw_Servo_Motor.Control_Angle(
                    Claw_Servo_Motor_Target_Angle,
                    500
                );


            if (tx_status == HAL_OK)
            {
                Complete_Outputs();
            }


            break;
        }


        // ====================================================
        // TOF闭环下降
        //
        // 非常重要：
        //
        // 这里不再发送固定-5°。
        //
        // Lift控制全部由：
        //
        // TIM_1ms_PeriodElapsedCallback()
        //
        // 负责。
        // ====================================================

        case Enum_Grap_Move_Descend:
        {
            break;
        }


        // ====================================================
        // Claw正式夹紧
        // ====================================================

        case Enum_Grap_Move_Clip:
        {
            if (Grap_Output_Flag)
            {
                break;
            }


            Claw_Servo_Motor_Target_Angle =
                Claw_Servo_Motor_Angle_Clip;


            tx_status =
                Claw_Servo_Motor.Control_Angle(
                    Claw_Servo_Motor_Target_Angle,
                    500
                );


            if (tx_status == HAL_OK)
            {
                Complete_Outputs();
            }


            break;
        }


        // ====================================================
        // Lift重新抬升
        // ====================================================

        case Enum_Grap_Move_Lift:
        {
            if (Grap_Output_Flag)
            {
                break;
            }


            Lift_Servo_Motor_Target_Angle =
                Lift_Servo_Motor_Angle_Lift;


            tx_status =
                Lift_Servo_Motor.Control_Angle(
                    Lift_Servo_Motor_Target_Angle,
                    500
                );


            if (tx_status == HAL_OK)
            {
                Complete_Outputs();
            }


            break;
        }


        // ====================================================
        // X/Y/Yaw去识别托盘
        // ====================================================

        case Enum_Grap_Move_X_Y_Tray:
        {
            // ------------------------------------------------
            // Step 0
            //
            // X/Y
            // ------------------------------------------------

            if (Grap_Output_Step == 0)
            {
                xy_status = XY_Motion.Move_To_mm(
                    Tray_X_mm, Tray_Y_mm, X_Move_Speed_mm_s, Y_Move_Speed_mm_s);

                if (xy_status == HAL_OK)
                {
                    Grap_Output_Step =
                        1;
                }
                else
                {
                    if (xy_status != HAL_BUSY)
                    {
                        Abort_XY_Fault();
                    }
                    break;
                }
            }


            // ------------------------------------------------
            // Step 1
            //
            // Yaw
            // ------------------------------------------------

            if (Grap_Output_Step == 1)
            {
                Yaw_Servo_Motor_Target_Angle =
                    Tray_Yaw_Angle;


                tx_status =
                    Yaw_Servo_Motor.Control_Angle(
                        Yaw_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        2;


                    Complete_Outputs();
                }


                break;
            }


            break;
        }


        // ====================================================
        // 识别托盘下降
        //
        // 托盘高度固定
        //
        // 所以这里仍然直接使用固定角度。
        // ====================================================

        case Enum_Grap_Move_Tray_Descend:
        {
            if (Grap_Output_Flag)
            {
                break;
            }


            Lift_Servo_Motor_Target_Angle =
                Lift_Servo_Motor_Angle_Recognition_Tray_Descend;


            tx_status =
                Lift_Servo_Motor.Control_Angle(
                    Lift_Servo_Motor_Target_Angle,
                    500
                );


            if (tx_status == HAL_OK)
            {
                Complete_Outputs();
            }


            break;
        }


        // ====================================================
        // 释放
        // ====================================================

        case Enum_Grap_Move_Release:
        {
            if (Grap_Output_Flag)
            {
                break;
            }


            Claw_Servo_Motor_Target_Angle =
                Claw_Servo_Motor_Angle_Release;


            tx_status =
                Claw_Servo_Motor.Control_Angle(
                    Claw_Servo_Motor_Target_Angle,
                    500
                );


            if (tx_status == HAL_OK)
            {
                Complete_Outputs();
            }


            break;
        }


        // ====================================================
        // 释放以后Lift抬升
        // ====================================================

        case Enum_Grap_Move_Tray_Lift:
        {
            if (Grap_Output_Flag)
            {
                break;
            }


            Lift_Servo_Motor_Target_Angle =
                Lift_Servo_Motor_Angle_Lift;


            tx_status =
                Lift_Servo_Motor.Control_Angle(
                    Lift_Servo_Motor_Target_Angle,
                    500
                );


            if (tx_status == HAL_OK)
            {
                Complete_Outputs();
            }


            break;
        }


        // ====================================================
        // 移动到识别姿态
        //
        // 识别姿态与Disable完全独立。
        // 这里统一控制两根步进轴和三个舵机。
        // ====================================================

        case Enum_Grap_Move_Recognition_Position:
        {
            // ------------------------------------------------
            // Step 0: X/Y
            // ------------------------------------------------

            if (Grap_Output_Step == 0)
            {
                xy_status = XY_Motion.Move_To_mm(
                    Recognition_X_mm, Recognition_Y_mm, X_Move_Speed_mm_s, Y_Move_Speed_mm_s);

                if (xy_status == HAL_OK)
                {
                    Grap_Output_Step =
                        1;
                }
                else
                {
                    if (xy_status != HAL_BUSY)
                    {
                        Abort_XY_Fault();
                    }
                    break;
                }
            }


            // ------------------------------------------------
            // Step 1: Yaw
            // ------------------------------------------------

            if (Grap_Output_Step == 1)
            {
                Yaw_Servo_Motor_Target_Angle =
                    Recognition_Yaw_Angle;


                tx_status =
                    Yaw_Servo_Motor.Control_Angle(
                        Yaw_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        2;
                }


                break;
            }


            // ------------------------------------------------
            // Step 2: Lift
            // ------------------------------------------------

            if (Grap_Output_Step == 2)
            {
                Lift_Servo_Motor_Target_Angle =
                    Recognition_Lift_Angle;


                tx_status =
                    Lift_Servo_Motor.Control_Angle(
                        Lift_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        3;
                }


                break;
            }


            // ------------------------------------------------
            // Step 3: Claw
            // ------------------------------------------------

            if (Grap_Output_Step == 3)
            {
                Claw_Servo_Motor_Target_Angle =
                    Recognition_Claw_Angle;


                tx_status =
                    Claw_Servo_Motor.Control_Angle(
                        Claw_Servo_Motor_Target_Angle,
                        500
                    );


                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step =
                        4;


                    Complete_Outputs();
                }


                break;
            }


            break;
        }


        // ====================================================
        // 回待机位置
        // ====================================================

        case Enum_Grap_Move_Return:
        {
            // TOF 失败可能在 Lift 仍下降时进入回程。先抬升 Lift，
            // 再移动 XY；在 XY 实际回到待机位置前保持夹爪闭合。
            if (Grap_Output_Step == 0)
            {
                Lift_Servo_Motor_Target_Angle = Lift_Servo_Motor_Angle_Disable;
                tx_status = Lift_Servo_Motor.Control_Angle(
                    Lift_Servo_Motor_Target_Angle, Servo_Move_Time_ms);
                if (tx_status == HAL_OK)
                {
                    Return_Lift_Command_Tick = HAL_GetTick();
                    Grap_Output_Step = 1;
                }
                break;
            }
            if (Grap_Output_Step == 1)
            {
                if ((uint32_t)(HAL_GetTick() - Return_Lift_Command_Tick) < Servo_Move_Time_ms)
                    break;
                Grap_Output_Step = 2;
            }
            if (Grap_Output_Step == 2)
            {
                xy_status = XY_Motion.Move_To_mm(
                    Disable_X_mm, Disable_Y_mm, X_Move_Speed_mm_s, Y_Move_Speed_mm_s);
                if (xy_status == HAL_OK) Grap_Output_Step = 3;
                else
                {
                    if (xy_status != HAL_BUSY) Abort_XY_Fault();
                    break;
                }
            }
            if (Grap_Output_Step == 3)
            {
                Yaw_Servo_Motor_Target_Angle = Yaw_Servo_Motor_Angle_Disable;
                tx_status = Yaw_Servo_Motor.Control_Angle(
                    Yaw_Servo_Motor_Target_Angle, Servo_Move_Time_ms);
                if (tx_status == HAL_OK) Grap_Output_Step = 4;
                break;
            }
            if (Grap_Output_Step == 4)
            {
                if (!XY_Move_Finished()) break;
                Claw_Servo_Motor_Target_Angle = Claw_Servo_Motor_Angle_Disable;
                tx_status = Claw_Servo_Motor.Control_Angle(
                    Claw_Servo_Motor_Target_Angle, Servo_Move_Time_ms);
                if (tx_status == HAL_OK)
                {
                    Grap_Output_Step = 5;
                    Complete_Outputs();
                }
            }
            break;
        }


        // ====================================================
        // FINISH
        // ====================================================

        case Enum_Grap_Move_Finish:
        case Enum_Grap_Move_Fault:
        {
            break;
        }


        default:
        {
            break;
        }
    }
}


// ============================================================
// Start Grap
// ============================================================

HAL_StatusTypeDef
Class_Holding_Jaw::Start_Grap(
    float Target_X_mm,
    float Target_Y_mm,
    float Target_Yaw_Angle,
    float Target_Preopen_mm,
    bool Target_Preopen_Valid)
{
    // ========================================================
    // 只有完整回到 Disable 后才允许开始下一件抓取。
    // 识别姿态 Finish 仍可能位于顶部相机视野内，不能直接起抓。
    // ========================================================

    if (Grap_Move_Status != Enum_Grap_Move_Disable)
    {
        return HAL_BUSY;
    }


    // ========================================================
    // X/Y必须建立坐标；未建立时返回明确错误，不伪装为暂时BUSY。
    // ========================================================

    if (!XY_Motion.Get_Position_Valid() || XY_Motion.Get_Fault())
    {
        Grap_Error = Enum_Grap_Error_XY_Position_Invalid;
        return HAL_ERROR;
    }

    if (!Motion_Configuration_Valid())
    {
        Grap_Error = Enum_Grap_Error_Target_Invalid;
        return HAL_ERROR;
    }

    // 所有待机舵机命令都必须经过完整动作时间，且实测 XY 必须位于待机坐标；
    // 独立标定可能改变当前 XY 位置，不能只看状态名。
    if (!Get_Disable_Ready()) return HAL_BUSY;


    // ========================================================
    // TOF必须在线并有有效距离
    //
    // 第一阶段调试建议保留。
    // ========================================================

    if (!TOF050F.Get_Distance_Valid())
    {
        Grap_Error = Enum_Grap_Error_TOF_Invalid;
        return HAL_ERROR;
    }


    // ========================================================
    // 机械工作区硬限位（与上位机配置一致）
    // ========================================================
    if (!std::isfinite(Target_X_mm) || !std::isfinite(Target_Y_mm) ||
        Target_X_mm < XY_Config::X_Min_mm || Target_X_mm > XY_Config::X_Max_mm ||
        Target_Y_mm < XY_Config::Y_Min_mm || Target_Y_mm > XY_Config::Y_Max_mm ||
        !std::isfinite(Target_Yaw_Angle) ||
        Target_Yaw_Angle < (-135.0f * PI / 180.0f) ||
        Target_Yaw_Angle > ( 135.0f * PI / 180.0f) ||
        (Target_Preopen_Valid &&
         (!std::isfinite(Target_Preopen_mm) ||
          Target_Preopen_mm <= 0.0f ||
          Target_Preopen_mm > 120.0f)))
    {
        Grap_Error = Enum_Grap_Error_Target_Invalid;
        return HAL_ERROR;
    }

    // 新夹爪使用 CAD 四连杆查表，基准为用户指定的 -15° 闭合角。
    // 修改目标和状态前先校验，不能回退到旧版固定 10° 角度。
    float preopen_deg = 0.0f;
    if (!Claw_Dynamic_Preopen_Enabled || !Target_Preopen_Valid ||
        !ClawPreopen::ToAngle(Target_Preopen_mm, preopen_deg))
    {
        Grap_Error = Enum_Grap_Error_Target_Invalid;
        return HAL_ERROR;
    }

    Grap_Error = Enum_Grap_Error_None;
    Grap_Failed = false;

    // ========================================================
    // 保存视觉目标
    // ========================================================

    Grap_Target_X_mm =
        Target_X_mm;


    Grap_Target_Y_mm =
        Target_Y_mm;


    Grap_Target_Yaw_Angle =
        Target_Yaw_Angle;

    Grap_Target_Preopen_mm = Target_Preopen_mm;
    Grap_Target_Preopen_Valid = Target_Preopen_Valid;

    // 使用统一线性公式完成“内侧开口量 -> 夹爪舵机角度”换算。
    Grap_Target_Pre_Clip_Angle = preopen_deg * PI / 180.0f;


    // ========================================================
    // 开始
    // ========================================================

    Set_Grap_Move_Status(
        Enum_Grap_Move_X_Y_Start
    );


    return HAL_OK;
}


// ============================================================
// 请求回到待机位置
// ============================================================

HAL_StatusTypeDef
Class_Holding_Jaw::Request_Return_To_Disable()
{
    if (Grap_Move_Status == Enum_Grap_Move_Disable)
    {
        if (Get_Disable_Ready()) return HAL_OK;
        // 上电 STOP 时可能还没有原点。未建立坐标但已经停稳的滑台，
        // 不能虚构当前位置，也不能因此强行驱动舵机或执行回位。
        if (XY_Motion.Get_Fault() || !XY_Motion.Get_Position_Valid()) return HAL_ERROR;
        if (!Motion_Configuration_Valid())
        {
            Grap_Error = Enum_Grap_Error_Target_Invalid;
            return HAL_ERROR;
        }
        if (!Get_Stop_Ready()) return HAL_BUSY;
        if (XY_At_mm(Disable_X_mm, Disable_Y_mm) && !Calibration_Yaw_Override) return HAL_BUSY;
        Set_Grap_Move_Status(Enum_Grap_Move_Return);
        return HAL_OK;
    }
    if (Grap_Move_Status == Enum_Grap_Move_Fault ||
        XY_Motion.Get_Fault() || !XY_Motion.Get_Position_Valid())
    {
        Abort_XY_Fault();
        return HAL_ERROR;
    }

    if (!Motion_Configuration_Valid())
    {
        Abort_Grap_Fault(Enum_Grap_Error_Target_Invalid);
        return HAL_ERROR;
    }

    // 调度器重复请求时，不要重置输出步骤和阶段超时计时。
    if (Grap_Move_Status == Enum_Grap_Move_Return) return HAL_OK;


    // 正常侧面识别期间通常停在Finish。
    // Recognition_Position是为了兼容STOP刚好发生在切换过程。
    if (Grap_Move_Status == Enum_Grap_Move_Finish ||
        Grap_Move_Status == Enum_Grap_Move_Recognition_Position)
    {
        Set_Grap_Move_Status(
            Enum_Grap_Move_Return
        );

        return HAL_OK;
    }


    return HAL_BUSY;
}


// ============================================================
// Set Grap Move Status
// ============================================================

void Class_Holding_Jaw::Set_Grap_Move_Status(
    Enum_Grap_Move_Status New_Status)
{
    Grap_State_Start_Tick = HAL_GetTick();
    Grap_Move_Status =
        New_Status;


    Grap_Move_Count =
        0;


    Grap_Output_Flag =
        false;


    Grap_Output_Step =
        0;


    // ========================================================
    // 进入TOF下降状态
    // ========================================================

    if (New_Status ==
        Enum_Grap_Move_Descend)
    {
        // ----------------------------------------------------
        // 从当前抬升位置开始
        //
        // -70°
        // ----------------------------------------------------

        Lift_Descend_Target_Angle =
            Lift_Servo_Motor_Angle_Lift;


        Lift_Servo_Motor_Target_Angle =
            Lift_Descend_Target_Angle;


        Lift_Control_Send_Count =
            0;


        Grap_TOF_Reached =
            false;


        Grap_Lift_Limit_Commanded =
            false;


        Grap_TOF_Stable_Count =
            0;


        Grap_TOF_Distance_mm =
            TOF050F.Get_Distance_mm();


        // ----------------------------------------------------
        // 很重要：
        //
        // 记录当前已经收到的最后一帧。
        //
        // 进入Descend以后，
        // 只统计“新的TOF帧”。
        //
        // 防止进入Descend前的旧距离
        // 被误认为新的距离数据。
        // ----------------------------------------------------

        Grap_TOF_Last_Rx_Count =
            TOF050F.Get_Data().Rx_Count;


        Grap_TOF_No_Update_Count_ms =
            0;
    }
}


// ============================================================
// XY Finished
// ============================================================

bool Class_Holding_Jaw::
XY_Move_Finished() const
{
    if (!XY_Motion.Get_Position_Valid() || XY_Motion.Get_Fault() ||
        !XY_Motion.Get_Motion_Finished()) return false;
    switch (Grap_Move_Status)
    {
        case Enum_Grap_Move_X_Y_Start: return XY_At_mm(Grap_Target_X_mm, Grap_Target_Y_mm);
        case Enum_Grap_Move_X_Y_Tray: return XY_At_mm(Tray_X_mm, Tray_Y_mm);
        case Enum_Grap_Move_Recognition_Position: return XY_At_mm(Recognition_X_mm, Recognition_Y_mm);
        case Enum_Grap_Move_Return: return XY_At_mm(Disable_X_mm, Disable_Y_mm);
        default: return true;
    }
}

bool Class_Holding_Jaw::XY_At_mm(float X_mm, float Y_mm) const
{
    const float x = XY_Motion.Get_X_mm(), y = XY_Motion.Get_Y_mm();
    return std::isfinite(x) && std::isfinite(y) &&
        std::fabs(x - X_mm) <= XY_Config::Position_Tolerance_mm &&
        std::fabs(y - Y_mm) <= XY_Config::Position_Tolerance_mm;
}

bool Class_Holding_Jaw::Get_Outputs_Settled() const
{
    return Grap_Output_Flag &&
        (uint32_t)(HAL_GetTick() - Grap_Output_Complete_Tick) >= Servo_Move_Time_ms;
}

bool Class_Holding_Jaw::Get_Disable_Ready() const
{
    return Grap_Move_Status == Enum_Grap_Move_Disable && !Calibration_Yaw_Override &&
        Motion_Configuration_Valid() && Get_Outputs_Settled() &&
        XY_Motion.Get_Position_Valid() && !XY_Motion.Get_Fault() &&
        XY_Motion.Get_Stop_Confirmed() && XY_At_mm(Disable_X_mm, Disable_Y_mm);
}

bool Class_Holding_Jaw::Get_Homing_Ready() const
{
    // 回零本身负责建立 Origin，因此这里不能调用 Get_Disable_Ready。
    return Grap_Move_Status == Enum_Grap_Move_Disable && Motion_Configuration_Valid() &&
        Get_Outputs_Settled() && !XY_Motion.Get_Fault() &&
        XY_Motion.Get_Stop_Confirmed();
}

bool Class_Holding_Jaw::Get_Stop_Ready() const
{
    if (!XY_Motion.Get_Stop_Confirmed()) return false;
    if (Grap_Move_Status == Enum_Grap_Move_Disable)
        return Get_Outputs_Settled() && (!Calibration_Yaw_Override ||
            (uint32_t)(HAL_GetTick() - Calibration_Yaw_Command_Tick) >= Calibration_Yaw_Settle_ms);
    return Grap_Move_Status == Enum_Grap_Move_Fault &&
        !Fault_Lift_Stop_Pending && !Fault_Yaw_Stop_Pending &&
        (uint32_t)(HAL_GetTick() - Fault_Stop_Complete_Tick) >= Servo_Stop_Settle_ms;
}

HAL_StatusTypeDef Class_Holding_Jaw::Control_Calibration_Yaw(float Radians, uint16_t Duration_ms)
{
    if (!std::isfinite(Radians) || Radians < -135.0f * PI / 180.0f ||
        Radians > 135.0f * PI / 180.0f || Duration_ms == 0U || Duration_ms > 9999U)
        return HAL_ERROR;
    if (Grap_Move_Status != Enum_Grap_Move_Disable || !Get_Stop_Ready()) return HAL_BUSY;
    const auto status = static_cast<HAL_StatusTypeDef>(Yaw_Servo_Motor.Control_Angle(Radians, Duration_ms));
    if (status == HAL_OK)
    {
        Yaw_Servo_Motor_Target_Angle = Radians;
        Calibration_Yaw_Override = true;
        Calibration_Yaw_Command_Tick = HAL_GetTick();
        Calibration_Yaw_Settle_ms = Duration_ms;
    }
    return status;
}

HAL_StatusTypeDef Class_Holding_Jaw::Stop_Calibration_Yaw()
{
    if (!Calibration_Yaw_Override) return HAL_OK;
    if (Grap_Move_Status != Enum_Grap_Move_Disable && Grap_Move_Status != Enum_Grap_Move_Fault)
        return HAL_BUSY;
    const auto status = static_cast<HAL_StatusTypeDef>(Yaw_Servo_Motor.Stop());
    if (status == HAL_OK)
    {
        Calibration_Yaw_Command_Tick = HAL_GetTick();
        Calibration_Yaw_Settle_ms = Servo_Stop_Settle_ms;
    }
    return status;
}

void Class_Holding_Jaw::Complete_Outputs()
{
    Grap_Output_Flag = true;
    Grap_Output_Complete_Tick = HAL_GetTick();
}

bool Class_Holding_Jaw::Motion_Configuration_Valid() const
{
    const auto in_workspace = [](float x, float y)
    {
        return std::isfinite(x) && std::isfinite(y) &&
            x >= XY_Config::X_Min_mm && x <= XY_Config::X_Max_mm &&
            y >= XY_Config::Y_Min_mm && y <= XY_Config::Y_Max_mm;
    };
    const auto angle_valid = [](float angle)
    {
        return std::isfinite(angle) && angle >= -135.0f * PI / 180.0f &&
            angle <= 135.0f * PI / 180.0f;
    };
    return std::isfinite(X_Move_Speed_mm_s) && X_Move_Speed_mm_s > 0.0f &&
        std::isfinite(Y_Move_Speed_mm_s) && Y_Move_Speed_mm_s > 0.0f &&
        in_workspace(Tray_X_mm, Tray_Y_mm) && in_workspace(Recognition_X_mm, Recognition_Y_mm) &&
        in_workspace(Disable_X_mm, Disable_Y_mm) && angle_valid(Tray_Yaw_Angle) &&
        angle_valid(Recognition_Yaw_Angle) && angle_valid(Recognition_Lift_Angle) &&
        angle_valid(Recognition_Claw_Angle) && angle_valid(Yaw_Servo_Motor_Angle_Disable) &&
        angle_valid(Lift_Servo_Motor_Angle_Disable) && angle_valid(Claw_Servo_Motor_Angle_Disable) &&
        angle_valid(Lift_Servo_Motor_Angle_Recognition_Tray_Descend) &&
        angle_valid(Claw_Servo_Motor_Angle_Clip) && angle_valid(Claw_Servo_Motor_Angle_Release) &&
        angle_valid(Lift_Servo_Motor_Angle_Lift) && angle_valid(Lift_Descend_Max_Angle) &&
        Lift_Servo_Motor_Angle_Lift <= Lift_Descend_Max_Angle &&
        std::isfinite(Lift_Descend_Speed) && Lift_Descend_Speed > 0.0f;
}

void Class_Holding_Jaw::Abort_XY_Fault()
{
    Abort_Grap_Fault(Enum_Grap_Error_XY_Drive_Fault);
}

void Class_Holding_Jaw::Abort_Grap_Fault(Enum_Grap_Error Reason)
{
    if (Grap_Move_Status == Enum_Grap_Move_Fault)
    {
        return;
    }
    XY_Motion.Stop();
    // 故障回程时保持夹爪闭合，绝不能在工作区上方松爪。
    Fault_Lift_Stop_Pending = true;
    Fault_Yaw_Stop_Pending = true;
    Grap_Error = Reason;
    Grap_Failed = true;
    Set_Grap_Move_Status(Enum_Grap_Move_Fault);
    Service_Fault_Servo_Stop();
}

void Class_Holding_Jaw::Service_Fault_Servo_Stop()
{
    // 两个舵机共用 UART3，同一时刻最多只能有一条 DMA 发送成功。
    // 遇到 HAL_BUSY 时保留待发送请求，不能丢失 Yaw 停机命令。
    if (Fault_Lift_Stop_Pending)
    {
        if (Lift_Servo_Motor.Stop() == HAL_OK)
        {
            Fault_Lift_Stop_Pending = false;
            Fault_Stop_Complete_Tick = HAL_GetTick();
        }
        return;
    }
    if (Fault_Yaw_Stop_Pending && Yaw_Servo_Motor.Stop() == HAL_OK)
    {
        Fault_Yaw_Stop_Pending = false;
        Fault_Stop_Complete_Tick = HAL_GetTick();
    }
}

HAL_StatusTypeDef Class_Holding_Jaw::Confirm_XY_Origin_mm(float X_mm, float Y_mm)
{
    if (Grap_Move_Status != Enum_Grap_Move_Disable &&
        Grap_Move_Status != Enum_Grap_Move_Fault)
    {
        return HAL_BUSY;
    }
    if (Grap_Move_Status == Enum_Grap_Move_Fault && !Get_Stop_Ready()) return HAL_BUSY;
    const HAL_StatusTypeDef status = XY_Motion.Confirm_Origin_mm(X_mm, Y_mm);
    if (status == HAL_OK)
    {
        Grap_Error = Enum_Grap_Error_None;
        Grap_Failed = false;
        Set_Grap_Move_Status(Enum_Grap_Move_Disable);
    }
    return status;
}

HAL_StatusTypeDef Class_Holding_Jaw::Start_XY_Collision_Homing()
{
    // 自动回零不能要求已有原点，否则会形成无法启动的循环。
    if (!Get_Homing_Ready()) return HAL_BUSY;
    return XY_Motion.Start_Collision_Homing();
}

void Class_Holding_Jaw::Stop_XY_Collision_Homing()
{
    XY_Motion.Stop_Collision_Homing();
}
