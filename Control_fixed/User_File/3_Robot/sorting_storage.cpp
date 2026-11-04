//
// Created for Smart Sorting System
// Rewritten following holding_jaw style
//

#include "3_Robot/sorting_storage.h"

#include <cmath>


// ============================================================
// Init
// ============================================================

void Class_Sorting_Storage::Init(
    const FDCAN_HandleTypeDef *__DM_CAN_Handler,

    const uint8_t __DM_CAN_Rx_ID,

    const uint8_t __DM_CAN_Tx_ID,

    UART_HandleTypeDef *__Servo_UART_Handler,

    const uint8_t __Servo_ID)
{
    // ========================================================
    // 保存硬件句柄
    // ========================================================

    DM_CAN_Handler =
        __DM_CAN_Handler;


    Servo_UART_Handler =
        __Servo_UART_Handler;


    DM4310_CAN_Rx_ID =
        __DM_CAN_Rx_ID;


    // ========================================================
    // DM4310
    //
    // MIT模式：STM32自己做位置-速度双环PID，只下发力矩。
    // ========================================================

    DM4310.Init(
        DM_CAN_Handler,

        __DM_CAN_Rx_ID,

        __DM_CAN_Tx_ID,

        Motor_DM_Control_Method_NORMAL_MIT,

        12.5f,

        10.0f,

        10.0f,

        10.0f
    );


    // ========================================================
    // 初始只保存待机目标
    //
    // 非常重要：
    // 此时还没有收到真实电机反馈，
    // 所以暂时不开始2ms力矩闭环。
    // ========================================================

    DM4310_Target_Angle =
        DM4310_Disable_Angle;


    // MIT内部位置/速度PD全部关闭，只保留Torque通道。
    DM4310.Set_Control_Angle(0.0f);
    DM4310.Set_Control_Omega(0.0f);
    DM4310.Set_K_P(0.0f);
    DM4310.Set_K_D(0.0f);
    DM4310.Set_Control_Torque(0.0f);


    Reset_DM4310_PID();


    DM4310_Startup_Ready =
        false;

    DM4310_Startup_Count_100ms = 0U;
    DM4310_Xout_Last_Update_Count = 0U;
    DM4310_Xout_Stale_2ms_Count = 0U;
    Xout_Clock_Valid = Feedback_Clock_Valid = false;
    Xout_Seen_Count = Feedback_Seen_Count = Stable_Xout_Last_Count = 0U;
    Fault_Latched = false;
    Fault_Recovery_Requested = false;


    // ========================================================
    // 【关键修改】上电立刻清错 + 使能一次
    // 后面 100ms Alive 还会继续重试
    // ========================================================

    HAL_Delay(50);                      // 给电机驱动一点稳定时间
    DM4310.CAN_Send_Clear_Error();
    HAL_Delay(20);
    DM4310.CAN_Send_Enter();

    // ========================================================
    // 托盘舵机
    // ========================================================

    Tray_Flip_Servo_Motor.Init(
        Servo_UART_Handler,

        __Servo_ID,

        Servo_Motor_Work_Mode_270_CW
    );


    Tray_Flip_Servo_Motor.Set_Target_PWM(
        Tray_Servo_PWM_Disable
    );


    Tray_Last_Command_PWM =
        0;


    Tray_Last_Command_Valid =
        false;


    // ========================================================
    // 状态变量
    // ========================================================

    Target_Box_ID =
        0;


    DM4310_Stable_Count =
        0;


    Sort_Finished_Pending =
        false;


    Set_Storage_Status(
        Sorting_Storage_Status_WAIT_DM_READY
    );
}


// ============================================================
// 10ms Callback
//
// 和Holding_Jaw一致：
// 先Outputs，再Storage_Move
// ============================================================

void Class_Sorting_Storage::
TIM_10ms_PeriodElapsedCallback()
{
    Observe_DM4310_Feedback();
    if (DM4310_Startup_Ready && Storage_Status != Sorting_Storage_Status_ERROR &&
        Storage_Status != Sorting_Storage_Status_STOP_TRAY_RETURN &&
        !(Fault_Latched && Fault_Recovery_Requested &&
          Storage_Status == Sorting_Storage_Status_STOP_DM_RETURN) &&
        !DM4310_Feedback_Fresh())
    {
        Enter_Error();
    }
    if (!Storage_Output_Flag && Storage_Status != Sorting_Storage_Status_DISABLE &&
        Storage_Status != Sorting_Storage_Status_ERROR &&
        (uint32_t)(HAL_GetTick() - State_Start_Tick) >= Output_Submit_Timeout_ms)
    {
        Enter_Error();
    }
    // ========================================================
    // 每10ms读取一次0x51 xout。
    //
    // 普通反馈POS是p_m（转子折算位置），储物转盘的机械零位/盒位
    // 必须以输出轴编码器xout为准。
    // ========================================================

    DM4310.CAN_Send_Read_Output_Shaft_Angle();

    // ========================================================
    // 先执行当前状态输出
    // ========================================================

    Outputs();


    // ========================================================
    // 再判断当前状态是否完成
    // ========================================================

    Storage_Move();
}


// ============================================================
// 输出当前储物机构状态指令
// ============================================================

void Class_Sorting_Storage::Outputs()
{
    switch (Storage_Status)
    {
        // ====================================================
        // 上电等待DM
        //
        // 托盘只需要回正常位置一次。
        // ====================================================

        case Sorting_Storage_Status_WAIT_DM_READY:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            if (
                Try_Send_Tray_PWM(
                    Tray_Servo_PWM_Disable,

                    Tray_Return_Move_Time_ms
                )
            )
            {
                Storage_Output_Flag =
                    true;
            }


            break;
        }


        // ====================================================
        // 上电以后DM回待机位置
        // ====================================================

        case Sorting_Storage_Status_DM_STARTUP_RETURN:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            Set_DM4310_Target(
                DM4310_Target_Angle
            );


            Storage_Output_Flag =
                true;


            break;
        }


        // ====================================================
        // DISABLE
        //
        // 不在这里再次发送托盘回位命令。
        //
        // 原来的重复动作就是：
        // TRAY_RETURN已经回位一次，
        // 进入DISABLE后又发一次同样的PWM。
        //
        // 现在DISABLE只保持已有目标，不重复输出。
        // ====================================================

        case Sorting_Storage_Status_DISABLE:
        {
            Storage_Output_Flag =
                true;


            break;
        }


        // ====================================================
        // DM4310转目标盒
        // ====================================================

        case Sorting_Storage_Status_DM_MOVE_TO_BOX:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            Set_DM4310_Target(
                DM4310_Target_Angle
            );


            Storage_Output_Flag =
                true;


            break;
        }


        // ====================================================
        // 托盘翻转
        // ====================================================

        case Sorting_Storage_Status_TRAY_FLIP:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            if (
                Try_Send_Tray_PWM(
                    Tray_Servo_PWM_Flip,

                    Tray_Flip_Move_Time_ms
                )
            )
            {
                Storage_Output_Flag =
                    true;
            }


            break;
        }


        // ====================================================
        // 翻转保持
        // ====================================================

        case Sorting_Storage_Status_TRAY_HOLD:
        {
            Storage_Output_Flag =
                true;


            break;
        }


        // ====================================================
        // 托盘恢复正常位置
        // ====================================================

        case Sorting_Storage_Status_TRAY_RETURN:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            if (
                Try_Send_Tray_PWM(
                    Tray_Servo_PWM_Disable,

                    Tray_Return_Move_Time_ms
                )
            )
            {
                Storage_Output_Flag =
                    true;
            }


            break;
        }


        // ====================================================
        // DM4310回待机位置
        // ====================================================

        case Sorting_Storage_Status_DM_RETURN:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            Set_DM4310_Target(
                DM4310_Target_Angle
            );


            Storage_Output_Flag =
                true;


            break;
        }


        // ====================================================
        // STOP：先恢复托盘
        // ====================================================

        case Sorting_Storage_Status_STOP_TRAY_RETURN:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            if (
                Try_Send_Tray_PWM(
                    Tray_Servo_PWM_Disable,

                    Tray_Return_Move_Time_ms
                )
            )
            {
                Storage_Output_Flag =
                    true;
            }


            break;
        }


        // ====================================================
        // STOP：DM回待机位置
        // ====================================================

        case Sorting_Storage_Status_STOP_DM_RETURN:
        {
            if (Storage_Output_Flag)
            {
                break;
            }


            Set_DM4310_Target(
                DM4310_Target_Angle
            );


            Storage_Output_Flag =
                true;


            break;
        }


        // ====================================================
        // ERROR
        // ====================================================

        case Sorting_Storage_Status_ERROR:
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
// Storage Move
// ============================================================

void Class_Sorting_Storage::Storage_Move()
{
    switch (Storage_Status)
    {
        // ====================================================
        // 上电等待DM真正在线
        //
        // 这里解决：
        // MCU刚上电时Now_Angle还没有真实反馈，
        // 就直接按默认0计算/控制的问题。
        // ====================================================

        case Sorting_Storage_Status_WAIT_DM_READY:
        {
            if (
                !Fault_Latched && Tray_Disable_Ready() &&
                DM4310_Feedback_Fresh() && DM4310.Get_Status() ==
                    Motor_DM_Status_ENABLE
                &&
                DM4310.Get_Control_Status() ==
                    Motor_DM_Control_Status_ENABLE
                &&
                DM4310.Get_Output_Shaft_Angle_Valid()
            )
            {
                // --------------------------------------------
                // 位置闭环必须以0x51 xout真实输出轴编码器为准。
                //
                // 普通反馈POS是p_m，说明书把xout单独定义为
                // “采用电机输出轴编码器计算得出的输出轴位置”。
                // 因此不再用POS/PMAX去推断机械零位。
                // --------------------------------------------

                const float now_angle =
                    Get_DM4310_Output_Angle();

                const float wait_angle =
                    Normalize_Base_Angle(
                        DM4310_Disable_Angle
                    );

                const float circular_error =
                    Get_Circular_Angle_Error(
                        wait_angle,
                        now_angle
                    );

                DM4310_Startup_Ready =
                    true;

                DM4310_Xout_Last_Update_Count =
                    DM4310.Get_Output_Shaft_Update_Count();

                DM4310_Xout_Stale_2ms_Count =
                    0U;

                if (
                    std::fabs(circular_error)
                    <=
                    DM4310_Angle_Tolerance
                    && std::fabs(DM4310.Get_Now_Omega()) <= DM4310_Omega_Tolerance
                )
                {
                    // 真实输出轴已经在待机位，第二次上电不再动作。
                    DM4310_Target_Angle =
                        wait_angle;

                    Reset_DM4310_PID();

                    Set_Storage_Status(
                        Sorting_Storage_Status_DISABLE
                    );
                }
                else
                {
                    // 确实不在待机位，按真实输出轴最短圆周方向回零。
                    Set_DM4310_Target(
                        wait_angle
                    );

                    Set_Storage_Status(
                        Sorting_Storage_Status_DM_STARTUP_RETURN
                    );
                }
            }


            break;
        }


        // ====================================================
        // 上电后DM回待机位置
        // ====================================================

        case Sorting_Storage_Status_DM_STARTUP_RETURN:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            Update_DM4310_Stable_Count();


            if (
                DM4310_Stable_Count >=
                DM4310_Stable_Required
            )
            {
                Set_Storage_Status(
                    Sorting_Storage_Status_DISABLE
                );


                break;
            }


            if (
                Storage_Move_Count >=
                DM4310_Move_Timeout_Count
            )
            {
                Enter_Error();
            }


            break;
        }


        // ====================================================
        // DISABLE
        // ====================================================

        case Sorting_Storage_Status_DISABLE:
        {
            Storage_Move_Count =
                0;


            DM4310_Stable_Count =
                0;


            break;
        }


        // ====================================================
        // 1. DM4310转到目标储物盒
        // ====================================================

        case Sorting_Storage_Status_DM_MOVE_TO_BOX:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            Update_DM4310_Stable_Count();


            if (
                DM4310_Stable_Count >=
                DM4310_Stable_Required
            )
            {
                Set_Storage_Status(
                    Sorting_Storage_Status_TRAY_FLIP
                );


                break;
            }


            if (
                Storage_Move_Count >=
                DM4310_Move_Timeout_Count
            )
            {
                Enter_Error();
            }


            break;
        }


        // ====================================================
        // 2. 托盘翻转
        // ====================================================

        case Sorting_Storage_Status_TRAY_FLIP:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            if (
                Storage_Move_Count *
                    10U
                >=
                (
                    Tray_Flip_Move_Time_ms
                    +
                    Tray_Move_Safe_Time_ms
                )
            )
            {
                Set_Storage_Status(
                    Sorting_Storage_Status_TRAY_HOLD
                );
            }


            break;
        }


        // ====================================================
        // 3. 保持翻转
        // ====================================================

        case Sorting_Storage_Status_TRAY_HOLD:
        {
            Storage_Move_Count++;


            if (
                Storage_Move_Count *
                    10U
                >=
                Tray_Hold_Time_ms
            )
            {
                Set_Storage_Status(
                    Sorting_Storage_Status_TRAY_RETURN
                );
            }


            break;
        }


        // ====================================================
        // 4. 托盘回正常位置
        // ====================================================

        case Sorting_Storage_Status_TRAY_RETURN:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            if (
                Storage_Move_Count *
                    10U
                >=
                (
                    Tray_Return_Move_Time_ms
                    +
                    Tray_Move_Safe_Time_ms
                )
            )
            {
                // --------------------------------------------
                // 托盘真正恢复以后，
                // 再根据当前真实角度计算最近等效待机角。
                // --------------------------------------------

                Set_DM4310_Target(
                    DM4310_Disable_Angle
                );


                Set_Storage_Status(
                    Sorting_Storage_Status_DM_RETURN
                );
            }


            break;
        }


        // ====================================================
        // 5. DM4310回待机位置
        // ====================================================

        case Sorting_Storage_Status_DM_RETURN:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            Update_DM4310_Stable_Count();


            if (
                DM4310_Stable_Count >=
                DM4310_Stable_Required
            )
            {
                Sort_Finished_Pending =
                    true;


                Target_Box_ID =
                    0;


                Set_Storage_Status(
                    Sorting_Storage_Status_DISABLE
                );


                break;
            }


            if (
                Storage_Move_Count >=
                DM4310_Move_Timeout_Count
            )
            {
                Enter_Error();
            }


            break;
        }


        // ====================================================
        // STOP 1：先恢复托盘
        // ====================================================

        case Sorting_Storage_Status_STOP_TRAY_RETURN:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            if (
                Storage_Move_Count *
                    10U
                >=
                (
                    Tray_Return_Move_Time_ms
                    +
                    Tray_Move_Safe_Time_ms
                )
            )
            {
                if (Fault_Latched && Fault_Recovery_Requested)
                {
                    // An operator recovery may need several Alive periods to
                    // 清除驱动器故障并重新使能前，保持零力矩；
                    // 必须等托盘复位且实测反馈刷新后才能继续。
                    if (!DM4310_Feedback_Fresh())
                    {
                        if ((uint32_t)(HAL_GetTick() - State_Start_Tick) >= Output_Submit_Timeout_ms)
                            Enter_Error();
                        break;
                    }
                    DM4310_Startup_Ready = true;
                    DM4310_Xout_Last_Update_Count = DM4310.Get_Output_Shaft_Update_Count();
                    DM4310_Xout_Stale_2ms_Count = 0U;
                }
                if (DM4310_Startup_Ready)
                {
                    if (!DM4310_Feedback_Fresh())
                    {
                        Enter_Error();
                        break;
                    }
                    Set_DM4310_Target(
                        DM4310_Disable_Angle
                    );


                    Set_Storage_Status(
                        Sorting_Storage_Status_STOP_DM_RETURN
                    );
                }
                else
                {
                    if (Fault_Latched)
                    {
                        Enter_Error();
                        break;
                    }
                    Set_Storage_Status(
                        Sorting_Storage_Status_WAIT_DM_READY
                    );
                }
            }


            break;
        }


        // ====================================================
        // STOP 2：DM回待机
        // ====================================================

        case Sorting_Storage_Status_STOP_DM_RETURN:
        {
            if (!Storage_Output_Flag)
            {
                Storage_Move_Count =
                    0;


                break;
            }


            Storage_Move_Count++;


            Update_DM4310_Stable_Count();


            if (
                DM4310_Stable_Count >=
                DM4310_Stable_Required
            )
            {
                Target_Box_ID =
                    0;


                Set_Storage_Status(
                    Sorting_Storage_Status_DISABLE
                );


                break;
            }


            if (
                Storage_Move_Count >=
                DM4310_Move_Timeout_Count
            )
            {
                Enter_Error();
            }


            break;
        }


        // ====================================================
        // ERROR
        // ====================================================

        case Sorting_Storage_Status_ERROR:
        {
            break;
        }


        default:
        {
            Enter_Error();


            break;
        }
    }
}


// ============================================================
// 2ms Callback
//
// MIT Torque模式：
// 位置外环 -> 目标角速度
// 速度内环 -> 目标力矩
// 最终只把Torque写进MIT帧，电机内部Kp/Kd保持0。
// ============================================================

void Class_Sorting_Storage::
TIM_2ms_PeriodElapsedCallback()
{
    Observe_DM4310_Feedback();
    if (Storage_Status == Sorting_Storage_Status_ERROR)
    {
        Reset_DM4310_PID();
        DM4310.TIM_Send_PeriodElapsedCallback();
        return;
    }
    if (Fault_Latched && Fault_Recovery_Requested &&
        Storage_Status == Sorting_Storage_Status_STOP_TRAY_RETURN)
    {
        Reset_DM4310_PID();
        DM4310.TIM_Send_PeriodElapsedCallback();
        return;
    }
    // 上电还没有真实反馈以前，不允许力矩闭环开始工作。
    if (!DM4310_Startup_Ready)
    {
        return;
    }


    // 掉线/失能期间清掉PID状态。
    // 重新Enable后从0力矩重新爬升，避免旧积分突然打出去。
    if (
        DM4310.Get_Status() !=
            Motor_DM_Status_ENABLE
        ||
        DM4310.Get_Control_Status() !=
            Motor_DM_Control_Status_ENABLE
        ||
        !DM4310.Get_Output_Shaft_Angle_Valid()
        || !DM4310_Feedback_Fresh()
    )
    {
        Reset_DM4310_PID();
        DM4310.TIM_Send_PeriodElapsedCallback();
        return;
    }

    // --------------------------------------------------------
    // xout新鲜度保护
    // --------------------------------------------------------
    const uint32_t xout_update_count =
        DM4310.Get_Output_Shaft_Update_Count();

    if (
        xout_update_count !=
        DM4310_Xout_Last_Update_Count
    )
    {
        DM4310_Xout_Last_Update_Count =
            xout_update_count;

        DM4310_Xout_Stale_2ms_Count =
            0U;
    }
    else if (
        DM4310_Xout_Stale_2ms_Count <
        0xFFFFU
    )
    {
        DM4310_Xout_Stale_2ms_Count++;
    }

    if (
        DM4310_Xout_Stale_2ms_Count >
        DM4310_Xout_Stale_Limit_2ms
    )
    {
        // 输出轴位置超过100ms没有更新，禁止继续盲推力矩。
        Reset_DM4310_PID();
        DM4310.Set_Control_Torque(0.0f);
        DM4310.TIM_Send_PeriodElapsedCallback();
        return;
    }


    Update_DM4310_PID();


    DM4310
        .TIM_Send_PeriodElapsedCallback();
}


// ============================================================
// 100ms Alive
// ============================================================
//
// void Class_Sorting_Storage::
// TIM_100ms_Alive_PeriodElapsedCallback()
// {
//     DM4310
//         .TIM_Alive_PeriodElapsedCallback();
// }
void Class_Sorting_Storage::TIM_100ms_Alive_PeriodElapsedCallback()
{
    // 锁存故障不能因为总线上继续收到反馈就自动清除或重新使能。
    if (Fault_Latched)
    {
        // DM Alive 函数内部会自动发送清错/使能命令。在故障期间调用它会
        // latched fault would silently re-enable the mechanism without consent.
        if (Fault_Recovery_Requested &&
            (Storage_Status == Sorting_Storage_Status_STOP_TRAY_RETURN ||
             Storage_Status == Sorting_Storage_Status_STOP_DM_RETURN))
            DM4310.TIM_Alive_PeriodElapsedCallback();
        return;
    }
    DM4310.TIM_Alive_PeriodElapsedCallback();
    // ========================================================
    // 1. 上电启动阶段
    // ========================================================

    if (!DM4310_Startup_Ready)
    {
        DM4310_Startup_Count_100ms++;

        // 前 1 秒完全静默，让电机自己完成上电
        if (DM4310_Startup_Count_100ms < 10U)
        {
            return;
        }

        // ----------------------------------------------------
        // 已经收到有效反馈，且控制状态为 ENABLE → 启动成功
        // ----------------------------------------------------
        if (DM4310.Get_Status() == Motor_DM_Status_ENABLE &&
            DM4310.Get_Control_Status() == Motor_DM_Control_Status_ENABLE)
        {
            // 真正置位交给 Storage_Move 里的 WAIT_DM_READY 分支
            // 这里只负责持续发使能，不再阻塞
            return;
        }

        // ----------------------------------------------------
        // 还没起来：每 300ms 重发一轮 Clear + Enter
        // ----------------------------------------------------
        // 10 → 13 → 16 → ... 每 3 个 100ms 一次
        if (((DM4310_Startup_Count_100ms - 10U) % 3U) == 0U)
        {
            DM4310.CAN_Send_Clear_Error();
            DM4310_Startup_Clear_Count++;
        }
        else if (((DM4310_Startup_Count_100ms - 10U) % 3U) == 1U)
        {
            DM4310.CAN_Send_Enter();
            DM4310_Startup_Enter_Count++;
        }

        return;
    }

    // ========================================================
    // 2. 已完成上电同步以后，交回底层 Alive
    // ========================================================

    // Alive was already sampled above. Calling it twice in one 100ms period
    // 否则会重复比较同一个计数器，把仍在线的 DM 错误判为离线。
}

// ============================================================
// Start Sort
// ============================================================

HAL_StatusTypeDef
Class_Sorting_Storage::Start_Sort(
    const uint8_t Box_ID)
{
    // ========================================================
    // 箱号检查
    // ========================================================

    if (
        Box_ID < 1U
        ||
        Box_ID > 6U
    )
    {
        return HAL_ERROR;
    }

    if (Fault_Latched || Storage_Status == Sorting_Storage_Status_ERROR)
        return HAL_ERROR;


    // ========================================================
    // 必须已经完成上电同步
    // ========================================================

    if (!DM4310_Startup_Ready)
    {
        return HAL_BUSY;
    }


    // ========================================================
    // 当前必须空闲
    // ========================================================

    if (
        Storage_Status !=
        Sorting_Storage_Status_DISABLE
    )
    {
        return HAL_BUSY;
    }


    // ========================================================
    // DM必须在线且处于Enable
    // ========================================================

    if (
        DM4310.Get_Status() !=
            Motor_DM_Status_ENABLE
        ||
        DM4310.Get_Control_Status() !=
            Motor_DM_Control_Status_ENABLE
    )
    {
        return HAL_BUSY;
    }

    // 遇到暂时等待时，不要消费请求，也不要覆盖原目标。
    if (!Get_Disable_Ready()) return HAL_BUSY;


    // ========================================================
    // 保存目标
    // ========================================================

    Target_Box_ID =
        Box_ID;


    Sort_Finished_Pending =
        false;


    // ========================================================
    // 获取储物盒基础角
    // ========================================================

    const float base_angle =
        Box_Angle_Rad[
            Box_ID - 1U
        ];


    // ========================================================
    // 选择离当前位置最近的等效角
    // ========================================================

    Set_DM4310_Target(
        base_angle
    );


    // ========================================================
    // 开始
    // ========================================================

    Set_Storage_Status(
        Sorting_Storage_Status_DM_MOVE_TO_BOX
    );


    return HAL_OK;
}


// ============================================================
// 请求回到安全待机
// ============================================================

void Class_Sorting_Storage::Request_Disable()
{
    if (Fault_Latched) Fault_Recovery_Requested = true;
    Sort_Finished_Pending =
        false;


    Target_Box_ID =
        0;


    // ========================================================
    // 已经在待机
    //
    // 不重新发送舵机PWM，
    // 防止产生第二次回位动作。
    // ========================================================

    if (Storage_Status == Sorting_Storage_Status_STOP_TRAY_RETURN ||
        Storage_Status == Sorting_Storage_Status_STOP_DM_RETURN)
    {
        return;
    }

    if (Storage_Status == Sorting_Storage_Status_DISABLE && Get_Disable_Ready()) return;


    // ========================================================
    // DM尚未同步但托盘已安全回位时，保持等待即可。
    // 若调试接口曾把托盘翻开，STOP仍必须先恢复托盘。
    // ========================================================

    if (
        Storage_Status ==
        Sorting_Storage_Status_WAIT_DM_READY
        && Tray_Disable_Ready()
    )
    {
        return;
    }

    // 托盘回位期间，将取消的箱位动作冻结在当前实测角度。
    // 编码器反馈过期时，绝不能继续生成盲目目标或持续输出 PID 力矩。
    Observe_DM4310_Feedback();
    Reset_DM4310_PID();
    if (DM4310_Feedback_Fresh())
    {
        // An explicit recovery STOP may also finish a failed startup once
        // valid enabled feedback exists; it never sends Clear/Enter itself.
        DM4310_Startup_Ready = true;
        Set_DM4310_Target(Get_DM4310_Output_Angle());
    }


    // ========================================================
    // STOP安全逻辑
    //
    // 无论当前处于哪个动作状态：
    // 先回托盘，再回DM。
    // ========================================================

    Set_Storage_Status(
        Sorting_Storage_Status_STOP_TRAY_RETURN
    );
}


// ============================================================
// CAN 接收回调
// ============================================================

void Class_Sorting_Storage::
CAN_RxCpltCallback(
    FDCAN_HandleTypeDef *hfdcan,

    const FDCAN_RxHeaderTypeDef &Header)
{
    if (
        DM_CAN_Handler ==
            nullptr
        ||
        hfdcan !=
            DM_CAN_Handler
    )
    {
        return;
    }


    if (
        Header.IdType != FDCAN_STANDARD_ID ||
        Header.RxFrameType != FDCAN_DATA_FRAME ||
        Header.FDFormat != FDCAN_CLASSIC_CAN ||
        Header.DataLength != 8U ||
        (
            Header.Identifier
            &
            0x7FFU
        )
        !=
        DM4310_CAN_Rx_ID
    )
    {
        return;
    }


    DM4310
        .CAN_RxCpltCallback();
    Observe_DM4310_Feedback();
}


// ============================================================
// UART Callback
// ============================================================

void Class_Sorting_Storage::
UART_RxCpltCallback(
    UART_HandleTypeDef *huart,

    uint8_t *Buffer,

    uint16_t Length)
{
    if (
        Servo_UART_Handler ==
            nullptr
        ||
        huart !=
            Servo_UART_Handler
    )
    {
        return;
    }


    Tray_Flip_Servo_Motor
        .UART_RxCpltCallback(
            Buffer,

            Length
        );
}


// ============================================================
// Pop Finished
// ============================================================

bool Class_Sorting_Storage::
Pop_Sort_Finished()
{
    if (!Sort_Finished_Pending)
    {
        return false;
    }


    Sort_Finished_Pending =
        false;


    return true;
}


// ============================================================
// Set Box Angle
// ============================================================

void Class_Sorting_Storage::
Set_Box_Angle(
    const uint8_t Box_ID,

    const float Angle_Rad)
{
    if (
        Box_ID < 1U
        ||
        Box_ID > 6U || !std::isfinite(Angle_Rad)
    )
    {
        return;
    }


    Box_Angle_Rad[
        Box_ID - 1U
    ] =
        Normalize_Base_Angle(Angle_Rad);
}


// ============================================================
// Servo Debug
// ============================================================

HAL_StatusTypeDef
Class_Sorting_Storage::Debug_Set_Tray_PWM(
    const uint16_t PWM,

    const uint16_t Move_Time_ms)
{
    if (PWM < 500U || PWM > 2500U || Servo_UART_Handler == nullptr)
        return HAL_ERROR;
    // ========================================================
    // 自动流程运行时禁止人工调试抢占舵机
    // ========================================================

    if (
        Storage_Status !=
            Sorting_Storage_Status_DISABLE
        &&
        Storage_Status !=
            Sorting_Storage_Status_WAIT_DM_READY
    )
    {
        return HAL_BUSY;
    }


    // ========================================================
    // 强制发送
    //
    // 调试时即使PWM与上一条完全相同，
    // 也允许重新发一遍。
    // ========================================================

    if (
        Try_Send_Tray_PWM(
            PWM,

            Move_Time_ms,

            true
        )
    )
    {
        // ----------------------------------------------------
        // 如果此时还在WAIT_DM_READY，
        // 防止下一次10ms Outputs又立刻发送默认900覆盖调试值。
        // ----------------------------------------------------

        if (
            Storage_Status ==
            Sorting_Storage_Status_WAIT_DM_READY
        )
        {
            Storage_Output_Flag =
                true;
        }


        return HAL_OK;
    }


    if (
        Servo_UART_Handler ==
            nullptr
    )
    {
        return HAL_ERROR;
    }


    return HAL_BUSY;
}


// ============================================================
// Debug To Disable
// ============================================================

HAL_StatusTypeDef
Class_Sorting_Storage::Debug_Tray_To_Disable(
    const uint16_t Move_Time_ms)
{
    return
        Debug_Set_Tray_PWM(
            Tray_Servo_PWM_Disable,

            Move_Time_ms
        );
}


// ============================================================
// Debug To Flip
// ============================================================

HAL_StatusTypeDef
Class_Sorting_Storage::Debug_Tray_To_Flip(
    const uint16_t Move_Time_ms)
{
    return
        Debug_Set_Tray_PWM(
            Tray_Servo_PWM_Flip,

            Move_Time_ms
        );
}


// ============================================================
// Set Storage Status
// ============================================================

void Class_Sorting_Storage::Set_Storage_Status(
    Enum_Sorting_Storage_Status New_Status)
{
    State_Start_Tick = HAL_GetTick();
    Stable_Xout_Last_Count = DM4310.Get_Output_Shaft_Update_Count();
    Storage_Status =
        New_Status;


    Storage_Move_Count =
        0;


    DM4310_Stable_Count =
        0;


    Storage_Output_Flag =
        false;
}


// ============================================================
// Update DM Stable Count
// ============================================================

void Class_Sorting_Storage::Observe_DM4310_Feedback()
{
    const uint32_t xout_count = DM4310.Get_Output_Shaft_Update_Count();
    if (xout_count != Xout_Seen_Count)
    {
        Xout_Seen_Count = xout_count;
        Xout_Last_Tick = HAL_GetTick();
        Xout_Clock_Valid = true;
    }
    const uint32_t feedback_count = DM4310.Get_Feedback_Update_Count();
    if (feedback_count != Feedback_Seen_Count)
    {
        Feedback_Seen_Count = feedback_count;
        Feedback_Last_Tick = DM4310.Get_Feedback_Timestamp_ms();
        Feedback_Clock_Valid = true;
    }
}

bool Class_Sorting_Storage::DM4310_Feedback_Fresh()
{
    Observe_DM4310_Feedback();
    const uint32_t now = HAL_GetTick();
    return Xout_Clock_Valid && Feedback_Clock_Valid &&
        DM4310.Get_Output_Shaft_Angle_Valid() && DM4310.Get_Feedback_Valid() &&
        (uint32_t)(now - Xout_Last_Tick) <= DM_Feedback_Timeout_ms &&
        (uint32_t)(now - Feedback_Last_Tick) <= DM_Feedback_Timeout_ms &&
        DM4310.Get_Status() == Motor_DM_Status_ENABLE &&
        DM4310.Get_Control_Status() == Motor_DM_Control_Status_ENABLE &&
        std::isfinite(DM4310.Get_Output_Shaft_Angle()) &&
        std::isfinite(DM4310.Get_Now_Omega());
}

bool Class_Sorting_Storage::Tray_Disable_Ready() const
{
    return Tray_Last_Command_Valid && Tray_Last_Command_PWM == Tray_Servo_PWM_Disable &&
        (uint32_t)(HAL_GetTick() - Tray_Last_Command_Tick) >=
            (uint32_t)Tray_Last_Command_Move_Time_ms + Tray_Move_Safe_Time_ms;
}

bool Class_Sorting_Storage::Get_Disable_Ready()
{
    return Storage_Status == Sorting_Storage_Status_DISABLE && DM4310_Startup_Ready &&
        DM4310_Feedback_Fresh() && Tray_Disable_Ready() &&
        std::fabs(Get_Circular_Angle_Error(DM4310_Disable_Angle,Get_DM4310_Output_Angle())) <= DM4310_Angle_Tolerance &&
        std::fabs(DM4310.Get_Now_Omega()) <= DM4310_Omega_Tolerance;
}

bool Class_Sorting_Storage::Clear_Fault()
{
    if (!Get_Disable_Ready()) return false;
    Fault_Latched = false;
    Fault_Recovery_Requested = false;
    return true;
}

void Class_Sorting_Storage::
Update_DM4310_Stable_Count()
{
    const uint32_t count = DM4310.Get_Output_Shaft_Update_Count();
    if (!DM4310_Target_Reached())
    {
        DM4310_Stable_Count = 0U;
        Stable_Xout_Last_Count = count;
        return;
    }
    if (count == Stable_Xout_Last_Count) return;
    Stable_Xout_Last_Count = count;
    if (DM4310_Stable_Count < 255U)
    {
        DM4310_Stable_Count++;
    }
}


// ============================================================
// 设置 DM 目标角度
// ============================================================

void Class_Sorting_Storage::
Set_DM4310_Target(
    const float Target_Angle)
{
    // 转盘只关心“物理方向”，目标统一归一化到[0, 2π)。
    DM4310_Target_Angle =
        Normalize_Base_Angle(
            Target_Angle
        );


    Reset_DM4310_PID(
        false
    );
}


// ============================================================
// 判断 DM 是否到达目标
// ============================================================

bool Class_Sorting_Storage::
DM4310_Target_Reached()
{
    if (!DM4310_Feedback_Fresh()) return false;
    // ========================================================
    // 必须已经完成上电同步
    // ========================================================

    if (!DM4310_Startup_Ready)
    {
        return false;
    }


    // ========================================================
    // 必须在线
    // ========================================================

    if (
        DM4310.Get_Status() !=
        Motor_DM_Status_ENABLE
    )
    {
        return false;
    }


    // ========================================================
    // 驱动器必须处于Enable
    // ========================================================

    if (
        DM4310.Get_Control_Status() !=
        Motor_DM_Control_Status_ENABLE
    )
    {
        return false;
    }


    // ========================================================
    // 角度 + 速度同时满足
    // ========================================================

    const float angle_error =
        std::fabs(
            Get_Circular_Angle_Error(
                DM4310_Target_Angle,
                Get_DM4310_Output_Angle()
            )
        );


    const float omega =
        std::fabs(
            DM4310.Get_Now_Omega()
        );


    return
        angle_error <=
            DM4310_Angle_Tolerance
        &&
        omega <=
            DM4310_Omega_Tolerance;
}


// ============================================================
// Circular angle helpers
//
// 储物转盘只关心0~360°的物理方向。
// 无论达妙反馈落在第几圈，都先按2π折回，再取[-π,+π]
// 最短角误差。这样可以从根本上避免“第二次上电又转一圈”。
// ============================================================

float Class_Sorting_Storage::
Normalize_Base_Angle(const float Angle_Rad) const
{
    if (!std::isfinite(Angle_Rad))
    {
        return 0.0f;
    }

    float result =
        std::fmod(
            Angle_Rad,
            TWO_PI
        );

    if (result < 0.0f)
    {
        result += TWO_PI;
    }

    return result;
}


float Class_Sorting_Storage::
Normalize_Angle_Error(
    const float Angle_Error_Rad) const
{
    if (!std::isfinite(Angle_Error_Rad))
    {
        return 0.0f;
    }

    float result =
        std::fmod(
            Angle_Error_Rad + PI,
            TWO_PI
        );

    if (result < 0.0f)
    {
        result += TWO_PI;
    }

    return result - PI;
}


float Class_Sorting_Storage::
Get_Circular_Angle_Error(
    const float Target_Angle_Rad,
    const float Now_Angle_Rad) const
{
    return
        Normalize_Angle_Error(
            Normalize_Base_Angle(
                Target_Angle_Rad
            )
            -
            Normalize_Base_Angle(
                Now_Angle_Rad
            )
        );
}


// ============================================================
// Real output-shaft angle
//
// xout(0x51)来自输出轴编码器，不再使用普通反馈POS作为机械位置。
// ============================================================

float Class_Sorting_Storage::
Get_DM4310_Output_Angle() const
{
    if (!DM4310.Get_Output_Shaft_Angle_Valid())
    {
        return 0.0f;
    }

    const float corrected =
        DM4310_Output_Direction
        *
        (
            DM4310.Get_Output_Shaft_Angle()
            -
            DM4310_Output_Zero_Offset
        );

    return Normalize_Base_Angle(corrected);
}


float Class_Sorting_Storage::Clamp_Float(
    const float Value,
    const float Min_Value,
    const float Max_Value)
{
    if (Value < Min_Value)
    {
        return Min_Value;
    }

    if (Value > Max_Value)
    {
        return Max_Value;
    }

    return Value;
}


// ============================================================
// Reset MIT double-loop PID
// ============================================================

void Class_Sorting_Storage::Reset_DM4310_PID(
    const bool Reset_Torque_Command)
{
    DM4310_Pos_Integral =
        0.0f;

    DM4310_Omega_Integral =
        0.0f;

    DM4310_Pre_Omega_Error =
        0.0f;

    DM4310_Pre_Omega_Error_Valid =
        false;

    DM4310_PID_Target_Omega =
        0.0f;

    if (Reset_Torque_Command)
    {
        DM4310_PID_Torque_Command =
            0.0f;
    }


    // MIT电机内部PD关闭，Torque由STM32双环PID生成。
    DM4310.Set_Control_Angle(0.0f);
    DM4310.Set_Control_Omega(0.0f);
    DM4310.Set_K_P(0.0f);
    DM4310.Set_K_D(0.0f);
    DM4310.Set_Control_Torque(
        DM4310_Output_Direction * DM4310_PID_Torque_Command
    );
}


// ============================================================
// MIT double-loop PID
// ============================================================

void Class_Sorting_Storage::Update_DM4310_PID()
{
    const float now_angle =
        Get_DM4310_Output_Angle();

    const float now_omega =
        DM4310_Output_Direction * DM4310.Get_Now_Omega();


    // --------------------------------------------------------
    // 1. 位置外环
    //
    // angle_error严格限制到[-π,+π]，因此任何时候都只走最短方向。
    // 这同时屏蔽了上电后整圈编号/扣圈变化。
    // --------------------------------------------------------

    const float angle_error =
        Get_Circular_Angle_Error(
            DM4310_Target_Angle,
            now_angle
        );


    DM4310_Pos_Integral +=
        angle_error
        *
        DM4310_PID_DT;

    DM4310_Pos_Integral =
        Clamp_Float(
            DM4310_Pos_Integral,
            -DM4310_Pos_Integral_Limit,
            DM4310_Pos_Integral_Limit
        );


    const float pos_d =
        -now_omega;


    float target_omega =
        DM4310_Pos_Kp
        *
        angle_error
        +
        DM4310_Pos_Ki
        *
        DM4310_Pos_Integral
        +
        DM4310_Pos_Kd
        *
        pos_d;


    target_omega =
        Clamp_Float(
            target_omega,
            -DM4310_Move_Omega,
            DM4310_Move_Omega
        );


    DM4310_PID_Target_Omega =
        target_omega;


    // --------------------------------------------------------
    // 2. 速度内环
    // --------------------------------------------------------

    const float omega_error =
        target_omega
        -
        now_omega;


    DM4310_Omega_Integral +=
        omega_error
        *
        DM4310_PID_DT;

    DM4310_Omega_Integral =
        Clamp_Float(
            DM4310_Omega_Integral,
            -DM4310_Omega_Integral_Limit,
            DM4310_Omega_Integral_Limit
        );


    float omega_d =
        0.0f;

    if (DM4310_Pre_Omega_Error_Valid)
    {
        omega_d =
            (
                omega_error
                -
                DM4310_Pre_Omega_Error
            )
            /
            DM4310_PID_DT;
    }


    DM4310_Pre_Omega_Error =
        omega_error;

    DM4310_Pre_Omega_Error_Valid =
        true;


    float torque_target =
        DM4310_Omega_Kp
        *
        omega_error
        +
        DM4310_Omega_Ki
        *
        DM4310_Omega_Integral
        +
        DM4310_Omega_Kd
        *
        omega_d;


    torque_target =
        Clamp_Float(
            torque_target,
            -DM4310_PID_Torque_Limit,
            DM4310_PID_Torque_Limit
        );


    // --------------------------------------------------------
    // 3. 力矩斜坡
    // --------------------------------------------------------

    const float torque_delta =
        torque_target
        -
        DM4310_PID_Torque_Command;


    if (
        torque_delta
        >
        DM4310_PID_Torque_Slew_Per_2ms
    )
    {
        DM4310_PID_Torque_Command +=
            DM4310_PID_Torque_Slew_Per_2ms;
    }
    else if (
        torque_delta
        <
        -DM4310_PID_Torque_Slew_Per_2ms
    )
    {
        DM4310_PID_Torque_Command -=
            DM4310_PID_Torque_Slew_Per_2ms;
    }
    else
    {
        DM4310_PID_Torque_Command =
            torque_target;
    }


    // --------------------------------------------------------
    // 4. MIT只下发Torque
    // --------------------------------------------------------

    DM4310.Set_Control_Angle(0.0f);
    DM4310.Set_Control_Omega(0.0f);
    DM4310.Set_K_P(0.0f);
    DM4310.Set_K_D(0.0f);
    DM4310.Set_Control_Torque(
        DM4310_Output_Direction * DM4310_PID_Torque_Command
    );
}


// ============================================================
// Tray Servo Send
// ============================================================

bool Class_Sorting_Storage::
Try_Send_Tray_PWM(
    const uint16_t PWM,

    const uint16_t Move_Time_ms,

    const bool Force_Send)
{
    if (
        Servo_UART_Handler ==
        nullptr
    )
    {
        return false;
    }


    // ========================================================
    // 非强制模式：
    //
    // 如果上一条成功命令已经是相同PWM，
    // 直接视为输出完成，不再重复发送。
    //
    // 这就是修复“回位舵机连续动作两次”的核心。
    // ========================================================

    if (
        !Force_Send
        &&
        Tray_Last_Command_Valid
        &&
        Tray_Last_Command_PWM ==
            PWM
    )
    {
        Tray_Flip_Servo_Motor.Set_Target_PWM(
            PWM
        );


        return true;
    }


    // ========================================================
    // DMA / UART BUSY时不覆盖发送缓存
    //
    // 下一次周期继续重试。
    // ========================================================

    if (
        Servo_UART_Handler->gState !=
        HAL_UART_STATE_READY
    )
    {
        return false;
    }


    const uint8_t status =
        Tray_Flip_Servo_Motor
            .Control_PWM(
                PWM,

                Move_Time_ms
            );


    if (
        status ==
        HAL_OK
    )
    {
        Tray_Last_Command_PWM =
            Tray_Flip_Servo_Motor
                .Get_Target_PWM();


        Tray_Last_Command_Valid =
            true;

        Tray_Last_Command_Tick = HAL_GetTick();
        Tray_Last_Command_Move_Time_ms = Move_Time_ms;


        return true;
    }


    return false;
}


// ============================================================
// 故障处理
// ============================================================

void Class_Sorting_Storage::Enter_Error()
{
    Fault_Recovery_Requested = false;
    if (Storage_Status == Sorting_Storage_Status_ERROR) return;
    Fault_Latched = true;
    Sort_Finished_Pending =
        false;


    Target_Box_ID =
        0;


    // ========================================================
    // 故障时撤掉盲目的回零目标，立即清零本地PID力矩。
    // ========================================================

    // 故障处理必须停止本地力矩闭环；只有重新获得有效反馈后，
    // 才能通过显式 Request_Disable 请求安全回位。
    Reset_DM4310_PID();
    DM4310.TIM_Send_PeriodElapsedCallback();


    Set_Storage_Status(
        Sorting_Storage_Status_ERROR
    );
}
