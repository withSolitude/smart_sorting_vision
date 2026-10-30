//
// Created by Lenovo on 2026/1/12.
//

#include "ita_robot.h"

#include <cmath>


namespace
{
uint8_t Grap_Error_To_Sorting_Error_Code(
    const Enum_Grap_Error Error)
{
    switch (Error)
    {
        case Enum_Grap_Error_XY_Position_Invalid:
        case Enum_Grap_Error_XY_Drive_Fault:
        {
            return Sorting_Vision_Error_Grap_XY_Position_Invalid;
        }

        case Enum_Grap_Error_TOF_Invalid:
        {
            return Sorting_Vision_Error_Grap_TOF_Invalid;
        }

        case Enum_Grap_Error_Target_Invalid:
        {
            return Sorting_Vision_Error_Grap_Target_Invalid;
        }

        case Enum_Grap_Error_TOF_Timeout:
        {
            return Sorting_Vision_Error_Grap_TOF_Timeout;
        }

        case Enum_Grap_Error_None:
        default:
        {
            return Sorting_Vision_Error_Grap_Generic;
        }
    }
}
}


// ============================================================
// Init
// ============================================================

void Class_Robot::Init()
{
    // ========================================================
    // 原机械机构
    // ========================================================

    XY_Fault_Reported = false;
    Storage_Fault_Reported = false;
    Storage_Recovery_Requested = false;
    Vision_Link_Lost_Seconds = 0U;
    Vision_Link_Lost_Stop_Requested = false;
    Holding_Jaw.Init();
    Sorting_Storage.Init(&hfdcan1,0x01,0x00,&huart4,4);

    // ========================================================
    // 握手状态初始化
    //
    // 此时通信UART可能还没有绑定，
    // 但状态机先进入STOPPED。
    // ========================================================

    Sorting_Handshake.Init(
        &Sorting_Vision_UART
    );


    Pending_Pick_Request =
        Struct_Sorting_Pick_Request{};


    Pick_Start_Pending =
        false;
    Pick_Start_Wait_Count = 0U;


    Pick_Action_Active =
        false;


    Stop_Waiting_Safe_Disable =
        false;


    Sorting_Start_Request_Pending =
        false;


    Retry_Pick_Return_Active =
        false;


    Sort_Target_Box_ID =
        0;


    Sort_Action_Active =
        false;
    Sort_Done_Notify_Pending = false;


    Pending_Sort_Request = Struct_Sorting_Result_Request{};
    Sort_Start_Pending = false;
    Sort_Start_Wait_Count = 0;

    Storage_Test_Start_Pending = false;
    Storage_Test_Action_Active = false;
    Storage_Test_Done_Notify_Pending = false;
    Storage_Test_Box_ID = 0U;
    Storage_Test_Start_Wait_Count = 0U;

    Calib_Action_Active = false;
    Calib_Start_Pending = false;
    Calib_Target_X_mm = 0.0f;
    Calib_Target_Y_mm = 0.0f;
    Calib_Move_Count = 0;
    Calib_Stable_Count = 0;

    Yaw_Calib_Action_Active = false;
    Yaw_Calib_Start_Pending = false;
    Yaw_Stop_Pending = false;
    Yaw_Calib_Target_Deg = 0.0f;
    Yaw_Calib_Count_10ms = 0;


    Sort_Action_Simulation_Count =
        0;
}


// ============================================================
// 初始化与Jetson通信的UART
// ============================================================

void Class_Robot::
Init_Sorting_Communication(
    UART_HandleTypeDef *UART_Handler)
{
    Sorting_Vision_UART.Init(
        UART_Handler,
        0.1f
    );


    // UART重新初始化以后，
    // 同时重置握手状态。

    Sorting_Handshake.Init(
        &Sorting_Vision_UART
    );

    // UART/握手重新初始化后，清掉旧的实体START锁存，
    // 避免重连后无人工按键却自行启动。
    Sorting_Start_Request_Pending = false;
}


// ============================================================
// 实体短按START：只锁存请求，不要求此刻所有机构都Ready
// ============================================================

bool Class_Robot::
Request_Sorting_Start()
{
    // 未完成人工回零/标定前不能锁存 START，否则既可能阻止原点确认，
    // 也可能在确认后意外自动启动。
    if (!XY_Config::Mechanical_Calibration_Confirmed ||
        !Holding_Jaw.XY_Motion.Get_Position_Valid() ||
        Holding_Jaw.XY_Motion.Get_Fault() || Sorting_Storage.Get_Fault() ||
        Sorting_Handshake.Get_Storage_Test_In_Progress())
    {
        Sorting_Start_Request_Pending = false;
        return false;
    }

    // 运行中短按不能成为“下一轮自动再启动”的请求。
    if (Sorting_Handshake.Get_Running())
    {
        return false;
    }

    // Jetson上电同步STOP尚未完成时忽略START，
    // 防止“同步尚未READY就按键”在ACK后自动起动。
    if (Sorting_Handshake.Get_Stop_Ack_Pending())
    {
        return false;
    }

    // ERROR/标定/其他中间状态不接受START锁存。
    if (
        Sorting_Handshake.Get_State()
        !=
        Sorting_Handshake_State_STOPPED
    )
    {
        return false;
    }

    Sorting_Start_Request_Pending = true;
    return true;
}


// ============================================================
// START请求服务
//
// STOPPED短按以后，如果夹爪仍在回Disable、DM4310还在Startup、
// XY位置尚未有效等，Start_Sorting_Work()会暂时返回false。
// 请求保持锁存，10ms后继续尝试，直到真正启动成功。
// ============================================================

void Class_Robot::
Service_Sorting_Start_Request()
{
    if (!Sorting_Start_Request_Pending)
    {
        return;
    }

    if (Sorting_Handshake.Get_Running())
    {
        Sorting_Start_Request_Pending = false;
        return;
    }

    // 只允许在STOPPED等待Ready；ERROR等状态必须重新人工处理，
    // 不能保留一个旧START请求在故障恢复后突然自启动。
    if (
        Sorting_Handshake.Get_State()
        !=
        Sorting_Handshake_State_STOPPED
    )
    {
        Sorting_Start_Request_Pending = false;
        return;
    }

    // 标定运动结束后，夹爪状态名可能仍是 Disable，但实测滑台位于其他位置。
    // 必须先回到真实待机姿态，再允许顶部相机开始新一轮自动流程。
    if (!Holding_Jaw.Get_Disable_Ready() &&
        Holding_Jaw.XY_Motion.Get_Position_Valid() && !Sorting_Storage.Get_Busy())
    {
        if (Holding_Jaw.Request_Return_To_Disable() == HAL_ERROR)
        {
            Sorting_Start_Request_Pending = false;
            Sorting_Handshake.Notify_Error(Grap_Error_To_Sorting_Error_Code(Holding_Jaw.Get_Grap_Error()));
            return;
        }
    }

    if (Start_Sorting_Work())
    {
        Sorting_Start_Request_Pending = false;
    }
}


// ============================================================
// 统一START安全条件
// ============================================================

bool Class_Robot::
Is_Sorting_Start_Ready()
{
    return Is_Sorting_Stop_Ready() &&
        XY_Config::Mechanical_Calibration_Confirmed &&
        Holding_Jaw.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
        !Sorting_Handshake.Get_Storage_Test_In_Progress();
}

bool Class_Robot::Is_Sorting_Stop_Ready()
{
    return
        !Pick_Action_Active
        &&
        !Pick_Start_Pending
        &&
        !Stop_Waiting_Safe_Disable
        &&
        !Retry_Pick_Return_Active
        &&
        !Sort_Action_Active
        &&
        !Sort_Start_Pending
        &&
        !Sort_Done_Notify_Pending
        &&
        !Storage_Test_Start_Pending
        &&
        !Storage_Test_Action_Active
        &&
        !Storage_Test_Done_Notify_Pending
        &&
        !Calib_Action_Active
        &&
        !Yaw_Calib_Action_Active
        &&
        !Yaw_Stop_Pending
        &&
        Holding_Jaw.Get_Stop_Ready()
        &&
        Sorting_Storage.Get_Disable_Ready();
}


// ============================================================
// Jetson主动STOP安全ACK服务
// ============================================================

void Class_Robot::Service_Storage_Recovery()
{
    if (Storage_Recovery_Requested && Sorting_Storage.Get_Disable_Ready() &&
        Sorting_Storage.Clear_Fault())
    {
        Storage_Recovery_Requested = false;
        Storage_Fault_Reported = false;
    }
}

void Class_Robot::
Service_Sorting_Stop_Ready_Ack()
{
    if (!Sorting_Handshake.Get_Stop_Ack_Pending())
    {
        return;
    }

    if (!Is_Sorting_Stop_Ready())
    {
        return;
    }

    // Notify_Stop_Ready内部只有可靠TX队列真正接收ACK后才清pending。
    // 如果UART暂时忙，下一次10ms会继续尝试。
    Sorting_Handshake.Notify_Stop_Ready();
}


// ============================================================
// START
// ============================================================

bool Class_Robot::
Start_Sorting_Work()
{
    // ========================================================
    // 如果上一轮STOP以后夹爪仍然在安全回位，
    // 不允许马上重新启动。
    // ========================================================

    if (
        Sorting_Handshake.Get_Stop_Ack_Pending()
        ||
        !Is_Sorting_Start_Ready()
    )
    {
        return false;
    }


    return
        Sorting_Handshake
            .Request_Start_Work();
}


// ============================================================
// STOP
// ============================================================

bool Class_Robot::
Stop_Sorting_Work()
{
    // STOP（包括1.5s长按急停）必须取消尚未真正执行的START锁存。
    Sorting_Start_Request_Pending = false;

    return
        Sorting_Handshake
            .Request_Stop_Work();
}


// ============================================================
// 旧Toggle兼容接口
//
// 正式实体按键已经不再使用Toggle语义。为避免其它旧调用点误把
// “运行中短按”变成STOP，这里也改成安全语义：
// - STOPPED -> 锁存START；
// - RUNNING -> 忽略；
// - 其它状态 -> 拒绝。
// ============================================================

bool Class_Robot::
Toggle_Sorting_Work()
{
    if (Sorting_Handshake.Get_Running())
    {
        return false;
    }

    if (
        Sorting_Handshake.Get_State()
        ==
        Sorting_Handshake_State_STOPPED
    )
    {
        return Request_Sorting_Start();
    }

    return false;
}


// ============================================================
// 后续4310 + 托盘舵机完成时调用
// ============================================================

bool Class_Robot::
Notify_Sort_Mechanism_Done()
{
    if (!Sort_Action_Active || !Sort_Action_Simulation_Enable ||
        !Holding_Jaw.Get_Disable_Ready() || !Sorting_Storage.Get_Disable_Ready())
    {
        return false;
    }


    // 如果已经STOP，
    // 不再上报本件成功。

    if (
        !Sorting_Handshake
            .Get_Running()
    )
    {
        Sort_Action_Active =
            false;

        Sort_Start_Pending = false;
        Sort_Start_Wait_Count = 0;
        Pending_Sort_Request = Struct_Sorting_Result_Request{};

        if (Calib_Action_Active)
        {
            Holding_Jaw.XY_Motion.Stop();
            Calib_Action_Active = false;
            Calib_Move_Count = 0;
            Calib_Stable_Count = 0;
        }


        Sort_Target_Box_ID =
            0;


        return false;
    }


    const bool status =
        Sorting_Handshake
            .Notify_Sort_Done();


    if (status)
    {
        Sort_Action_Active =
            false;

        Sort_Start_Pending = false;
        Sort_Start_Wait_Count = 0;
        Pending_Sort_Request = Struct_Sorting_Result_Request{};

        if (Calib_Action_Active)
        {
            Holding_Jaw.XY_Motion.Stop();
            Calib_Action_Active = false;
            Calib_Move_Count = 0;
            Calib_Stable_Count = 0;
        }

        Sort_Target_Box_ID =
            0;


        Sort_Action_Simulation_Count =
            0;
    }


    return status;
}


// ============================================================
// Loop
// ============================================================

void Class_Robot::Loop()
{

}


// ============================================================
// 1000ms Alive
// ============================================================

void Class_Robot::
TIM_1000ms_Alive_PeriodElapsedCallback()
{
    Sorting_Vision_UART
        .TIM_1000ms_Alive_PeriodElapsedCallback();

    // 视觉链路掉线不等于某个机械动作超时。
    // 连续约3秒没有有效帧时，只安全停止当前分拣事务一次，不锁死整机；
    // 视觉端恢复后发送心跳/STOP，再由实体按键或上位机显式START继续。
    if (Sorting_Handshake.Get_Running() &&
        Sorting_Vision_UART.Get_Status() == Sorting_Vision_UART_Status_DISABLE)
    {
        if (Vision_Link_Lost_Seconds < 255U) ++Vision_Link_Lost_Seconds;
        if (Vision_Link_Lost_Seconds >= 3U && !Vision_Link_Lost_Stop_Requested)
        {
            Vision_Link_Lost_Stop_Requested = true;
            Sorting_Handshake.Request_Stop_Work();
        }
    }
    else if (Sorting_Vision_UART.Get_Status() == Sorting_Vision_UART_Status_ENABLE)
    {
        Vision_Link_Lost_Seconds = 0U;
        Vision_Link_Lost_Stop_Requested = false;
    }
}


// ============================================================
// 200ms
// ============================================================

void Class_Robot::
TIM_200ms_Alive_PeriodElapsedCallback()
{

}


// ============================================================
// 100ms Alive
// ============================================================

void Class_Robot::
TIM_100ms_Alive_PeriodElapsedCallback()
{
    Sorting_Storage
        .TIM_100ms_Alive_PeriodElapsedCallback();
}


// ============================================================
// 100ms机械状态机
// ============================================================

void Class_Robot::
TIM_100ms_Calculate_Callback()
{
    Holding_Jaw
        .TOF050F
        .TIM_100ms_PeriodElapsedCallback();


    Holding_Jaw
        .TIM_100ms_PeriodElapsedCallback();
}


// ============================================================
// STOP高优先级处理
//
// 必须在标定/PICK/SORT等新动作消费之前调用。现有正式语义保持为：
// - 尚未开始的任务立即取消；
// - 已开始的抓取允许安全放下并回Disable，不在半空直接开爪；
// - 标定XY立即停轴；
// - 储物机构请求回Disable。
// ============================================================
void Class_Robot::Handle_Stop_Request()
{
    bool explicit_recovery = false;
    if (!Sorting_Handshake.Pop_Stop_Request(&explicit_recovery))
    {
        return;
    }

    // Jetson/错误路径触发STOP时，也必须取消实体START锁存，
    // 否则机构回到Disable后可能出现非预期自动重启。
    Sorting_Start_Request_Pending = false;

    if (explicit_recovery) Storage_Recovery_Requested = true;
    // 机构故障不能自动授权重新使能转盘；必须由操作员或 Jetson 再次显式 STOP，
    // 才能请求恢复。
    if (!Sorting_Storage.Get_Fault() || explicit_recovery)
        Sorting_Storage.Request_Disable();

    Retry_Pick_Return_Active = false;

    Pick_Start_Pending = false;
    Pick_Start_Wait_Count = 0;
    Pending_Pick_Request = Struct_Sorting_Pick_Request{};

    if (Pick_Action_Active)
    {
        // 正在搬运物体时，不在半空强行回Disable。
        // 先让当前物体安全放到识别托盘并到达识别姿态，
        // 完成后再由下面的完成处理请求Return -> Disable。
        Stop_Waiting_Safe_Disable = true;
    }
    else
    {
        Stop_Waiting_Safe_Disable = false;

        // MCU 可能在 CAN 驱动器保留旧运动时复位。
        // Disable 状态没有 XY 回程任务，因此即使尚未建立原点，也要显式发送停机。
        if (Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Disable)
        {
            Holding_Jaw.XY_Motion.Stop();
        }

        // 正常流程完成后Holding Jaw会停在识别姿态(Finish)，
        // STOP/退出时才真正要求回Disable。
        if (!Calib_Action_Active && !Yaw_Calib_Action_Active && !Yaw_Stop_Pending)
            Holding_Jaw.Request_Return_To_Disable();
    }

    Sort_Action_Active = false;
    Sort_Start_Pending = false;
    Sort_Start_Wait_Count = 0;
    Sort_Done_Notify_Pending = false;
    Pending_Sort_Request = Struct_Sorting_Result_Request{};
    Sort_Target_Box_ID = 0;
    Sort_Action_Simulation_Count = 0;

    Storage_Test_Start_Pending = false;
    Storage_Test_Action_Active = false;
    Storage_Test_Done_Notify_Pending = false;
    Storage_Test_Start_Wait_Count = 0U;
    Storage_Test_Box_ID = 0U;

    if (Calib_Action_Active)
    {
        Holding_Jaw.XY_Motion.Stop();
        Calib_Action_Active = false;
        Calib_Start_Pending = false;
        Calib_Move_Count = 0;
        Calib_Stable_Count = 0;
    }

    if (Yaw_Calib_Action_Active)
    {
        // 立即请求停止当前Yaw舵机动作。若UART3恰好忙，握手状态仍已STOP，
        // 不会再启动任何后续标定/抓取动作。
        Yaw_Stop_Pending = true;
        Yaw_Calib_Action_Active = false;
        Yaw_Calib_Start_Pending = false;
        Yaw_Calib_Count_10ms = 0;
    }
}


// ============================================================
// 10ms
//
// 智能分拣总调度
// ============================================================

void Class_Robot::
TIM_10ms_Calculate_PeriodElapsedCallback()
{
    Sorting_Storage
    .TIM_10ms_PeriodElapsedCallback();
    // ========================================================
    // 1.
    // 处理Jetson串口命令
    // ========================================================

    Sorting_Handshake.Process();

    // 储物机构内部错误必须上抛，否则上位机会永久等待SORT_DONE。
    if (Sorting_Storage.Get_Fault() && !Storage_Fault_Reported)
    {
        Storage_Fault_Reported = true;
        Storage_Recovery_Requested = false;
        Sort_Action_Active = false;
        Sort_Start_Pending = false;
        Sort_Done_Notify_Pending = false;
        Pending_Sort_Request = Struct_Sorting_Result_Request{};

        Storage_Test_Start_Pending = false;
        Storage_Test_Action_Active = false;
        Storage_Test_Done_Notify_Pending = false;
        Storage_Test_Start_Wait_Count = 0U;
        Storage_Test_Box_ID = 0U;

        Sorting_Handshake.Notify_Error(3U);
    }

    Service_Storage_Recovery();

    // 反馈丢失、堵转或超时也会中止 STOP 回程和重试等待。
    // Failed axes cannot be driven to a supposed safe home without feedback.
    if (Service_XY_Fault())
    {
        Handle_Stop_Request();
        Service_Storage_Recovery();
        Service_Sorting_Stop_Ready_Ack();
        return;
    }

    // STOP/故障请求拥有最高机械调度优先级。
    Handle_Stop_Request();

    // Jetson主动STOP的ACK必须等所有机构真正安全回位后才发送。
    // 这条ACK同时作为上电同步完成边界。
    Service_Sorting_Stop_Ready_Ack();

    // 实体START请求采用锁存方式。
    // 如果上一轮STOP后机构还在回位，或DM4310尚未Ready，
    // 这里每10ms自动重试；Ready以后只真正START一次。
    Service_Sorting_Start_Request();


    // ========================================================
    // 储物机构独立测试
    //
    // 用于侧面识别 + 储物机构联调，不经过顶部抓取。
    // 仅在自动分拣 STOPPED 状态下接受。
    // ========================================================

    if (!Storage_Test_Action_Active &&
        !Storage_Test_Start_Pending &&
        !Storage_Test_Done_Notify_Pending)
    {
        uint8_t test_box_id = 0U;

        if (Sorting_Handshake.Pop_Storage_Test_Request(test_box_id))
        {
            const bool safe =
                Sorting_Handshake.Get_State() == Sorting_Handshake_State_STOPPED &&
                !Pick_Action_Active &&
                !Pick_Start_Pending &&
                !Stop_Waiting_Safe_Disable &&
                !Sort_Action_Active &&
                !Sort_Start_Pending &&
                !Calib_Action_Active &&
                !Yaw_Calib_Action_Active &&
                !Yaw_Stop_Pending &&
                Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Disable &&
                Holding_Jaw.Get_Stop_Ready() &&
                Sorting_Storage.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
                !Sorting_Handshake.Get_Stop_Ack_Pending();

            if (!safe)
            {
                Sorting_Handshake.Notify_Error(2U);
            }
            else
            {
                Storage_Test_Box_ID = test_box_id;
                Storage_Test_Start_Pending = true;
                Storage_Test_Start_Wait_Count = 0U;
            }
        }
    }

    if (Storage_Test_Start_Pending)
    {
        ++Storage_Test_Start_Wait_Count;

        const HAL_StatusTypeDef status =
            Sorting_Storage.Start_Sort(Storage_Test_Box_ID);

        if (status == HAL_OK)
        {
            Storage_Test_Start_Pending = false;
            Storage_Test_Start_Wait_Count = 0U;
            Storage_Test_Action_Active = true;
        }
        else if (status == HAL_ERROR ||
                 Storage_Test_Start_Wait_Count >=
                     Storage_Test_Start_Timeout_10ms)
        {
            Storage_Test_Start_Pending = false;
            Storage_Test_Start_Wait_Count = 0U;
            Storage_Test_Box_ID = 0U;
            Sorting_Handshake.Notify_Error(2U);
        }
        // HAL_BUSY：允许短暂等待DM完成启动/状态切换。
    }

    // 独立XY标定请求。只移动两根直线轴，不执行夹取/储物。
    if (!Calib_Action_Active)
    {
        Struct_Sorting_Calib_Request calib_request{};
        if (Sorting_Handshake.Pop_Calib_Request(calib_request))
        {
            const bool safe =
                Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Disable &&
                !Pick_Action_Active && !Pick_Start_Pending &&
                !Sort_Action_Active && !Sort_Start_Pending &&
                !Storage_Test_Start_Pending && !Storage_Test_Action_Active &&
                !Storage_Test_Done_Notify_Pending && !Yaw_Calib_Action_Active && !Yaw_Stop_Pending &&
                !Retry_Pick_Return_Active && !Stop_Waiting_Safe_Disable &&
                !Sort_Done_Notify_Pending && !Sorting_Handshake.Get_Stop_Ack_Pending() &&
                Sorting_Storage.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
                Holding_Jaw.Get_Outputs_Settled() &&
                XY_Config::Mechanical_Calibration_Confirmed &&
                Holding_Jaw.XY_Motion.Get_Position_Valid() &&
                !Holding_Jaw.XY_Motion.Get_Fault();

            if (!safe ||
                !std::isfinite(calib_request.X_mm) ||
                !std::isfinite(calib_request.Y_mm) ||
                calib_request.X_mm < XY_Config::X_Min_mm || calib_request.X_mm > XY_Config::X_Max_mm ||
                calib_request.Y_mm < XY_Config::Y_Min_mm || calib_request.Y_mm > XY_Config::Y_Max_mm)
            {
                Sorting_Handshake.Notify_Error(5U);
            }
            else
            {
                Calib_Action_Active = true;
                Calib_Start_Pending = true;
                Calib_Target_X_mm = calib_request.X_mm;
                Calib_Target_Y_mm = calib_request.Y_mm;
                Calib_Move_Count = 0;
                Calib_Stable_Count = 0;
            }
        }
    }

    if (Calib_Action_Active)
    {
        ++Calib_Move_Count;
        if (Calib_Start_Pending)
        {
            const HAL_StatusTypeDef status = Holding_Jaw.XY_Motion.Move_To_mm(
                Calib_Target_X_mm, Calib_Target_Y_mm,
                Calib_Move_Speed_mm_s, Calib_Move_Speed_mm_s);
            if (status == HAL_OK)
                Calib_Start_Pending = false;
            else if (status != HAL_BUSY || Calib_Move_Count >= Pick_Start_Timeout_10ms)
            {
                Calib_Action_Active = false;
                Calib_Start_Pending = false;
                Holding_Jaw.XY_Motion.Stop();
                Sorting_Handshake.Notify_Error(5U);
                Handle_Stop_Request();
                return;
            }
        }
        const float x_now = Holding_Jaw.XY_Motion.Get_X_mm();
        const float y_now = Holding_Jaw.XY_Motion.Get_Y_mm();
        const bool stopped = Holding_Jaw.XY_Motion.Get_Position_Valid() &&
            !Holding_Jaw.XY_Motion.Get_Fault() &&
            Holding_Jaw.XY_Motion.Get_Motion_Finished();
        const bool reached = !Calib_Start_Pending && stopped &&
            std::fabs(x_now - Calib_Target_X_mm) <= Calib_Position_Tolerance_mm &&
            std::fabs(y_now - Calib_Target_Y_mm) <= Calib_Position_Tolerance_mm;

        if (reached)
        {
            if (Calib_Stable_Count < 255U) ++Calib_Stable_Count;
            if (Calib_Stable_Count >= Calib_Stable_Required)
            {
                // 若TX队列暂时满，保持“已到位”状态，下个10ms只重试通知，
                // 不会重新执行机械运动。
                if (Sorting_Handshake.Notify_Calib_Reached(x_now, y_now))
                {
                    Calib_Action_Active = false;
                    Calib_Start_Pending = false;
                }
            }
        }
        else
        {
            Calib_Stable_Count = 0;
        }

        if (Calib_Action_Active && Calib_Move_Count >= Calib_Timeout_10ms)
        {
            Holding_Jaw.Abort_XY_Fault();
            Service_XY_Fault();
        }
    }


    // 独立Yaw标定请求。直接控制Yaw舵机角，不执行XY/下降/夹取/储物。
    if (!Yaw_Calib_Action_Active && !Calib_Action_Active)
    {
        Struct_Sorting_Yaw_Calib_Request yaw_request{};
        if (Sorting_Handshake.Pop_Yaw_Calib_Request(yaw_request))
        {
            const bool safe =
                Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Disable &&
                !Pick_Action_Active && !Pick_Start_Pending &&
                !Sort_Action_Active && !Sort_Start_Pending &&
                !Storage_Test_Start_Pending && !Storage_Test_Action_Active &&
                !Storage_Test_Done_Notify_Pending && !Sort_Done_Notify_Pending &&
                !Retry_Pick_Return_Active && !Stop_Waiting_Safe_Disable && !Yaw_Stop_Pending &&
                Sorting_Storage.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
                Holding_Jaw.Get_Stop_Ready() && !Sorting_Handshake.Get_Stop_Ack_Pending() &&
                std::isfinite(yaw_request.Servo_Command_Deg) &&
                yaw_request.Servo_Command_Deg >= Pick_Yaw_Servo_Command_Min_Deg &&
                yaw_request.Servo_Command_Deg <= Pick_Yaw_Servo_Command_Max_Deg;

            if (!safe)
            {
                Sorting_Handshake.Notify_Error(6U);
            }
            else
            {
                Yaw_Calib_Action_Active = true;
                Yaw_Calib_Start_Pending = true;
                Yaw_Calib_Target_Deg = yaw_request.Servo_Command_Deg;
                Yaw_Calib_Count_10ms = 0;
            }
        }
    }

    if (Yaw_Calib_Action_Active)
    {
        if (Yaw_Calib_Count_10ms < 65535U)
        {
            ++Yaw_Calib_Count_10ms;
        }

        if (Yaw_Calib_Start_Pending)
        {
            const HAL_StatusTypeDef status = Holding_Jaw.Control_Calibration_Yaw(
                Yaw_Calib_Target_Deg * PI / 180.0f, Yaw_Calib_Move_Time_ms);
            if (status == HAL_OK)
            {
                Yaw_Calib_Start_Pending = false;
                Yaw_Calib_Count_10ms = 0;
            }
            else if (status != HAL_BUSY || Yaw_Calib_Count_10ms >= Yaw_Calib_Timeout_10ms)
            {
                Yaw_Calib_Action_Active = false;
                Yaw_Calib_Start_Pending = false;
                Sorting_Handshake.Notify_Error(6U);
                Handle_Stop_Request();
                return;
            }
        }

        // 当前舵机驱动没有周期角度反馈，因此这里只能按设定运动时间+余量
        // 判断“允许人工测量”。真正机械角度由顶部相机两点测量得到。
        if (!Yaw_Calib_Start_Pending && Yaw_Calib_Count_10ms >= Yaw_Calib_Reach_Count_10ms)
        {
            if (Sorting_Handshake.Notify_Yaw_Calib_Reached(Yaw_Calib_Target_Deg))
            {
                Yaw_Calib_Action_Active = false;
                Yaw_Calib_Start_Pending = false;
                Yaw_Calib_Count_10ms = 0;
            }
        }

        if (Yaw_Calib_Action_Active &&
            Yaw_Calib_Count_10ms >= Yaw_Calib_Timeout_10ms)
        {
            Yaw_Calib_Action_Active = false;
            Yaw_Calib_Start_Pending = false;
            Yaw_Stop_Pending = true;
            Yaw_Calib_Count_10ms = 0;
            Sorting_Handshake.Notify_Error(6U);
        }
    }


    // STOP已在本周期最前面处理，禁止在标定/PICK/SORT之后再消费。


    // ========================================================
    // 独立 STORAGE_TEST 完成处理
    //
    // 重要：STORAGE_TEST 只允许在握手 STOPPED 状态运行，
    // 此时 Get_Running() == false。
    // 因此测试完成事件必须在下面的 !Get_Running() 提前 return 之前处理，
    // 否则机械动作虽然已经完成，MCU_STORAGE_TEST_DONE 永远不会发送。
    // ========================================================

    if (
        Storage_Test_Action_Active
        &&
        Sorting_Storage.Pop_Sort_Finished()
    )
    {
        Storage_Test_Action_Active = false;
        Storage_Test_Done_Notify_Pending = true;
    }

    if (Storage_Test_Done_Notify_Pending)
    {
        if (Sorting_Storage.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
            Sorting_Handshake.Notify_Storage_Test_Done(Storage_Test_Box_ID))
        {
            Storage_Test_Done_Notify_Pending = false;
            Storage_Test_Box_ID = 0U;
        }
    }


    // ========================================================
    // 3.
    // 判断当前第一阶段机械流程是否结束
    //
    // 一定放在是否Running判断前，
    // 因为STOP后夹爪仍然需要安全回Disable。
    // ========================================================

    if (
        Pick_Action_Active
        &&
        Holding_Jaw
            .Get_Grap_Finished()
    )
    {
        // ----------------------------------------------------
        // STOP过程中：
        //
        // 1) 如果当前正常流程已经完成并停在识别姿态(Finish)，
        //    先请求Return -> Disable；
        // 2) 真正到Disable以后才清掉STOP等待标志。
        //
        // 这样不会在半空开爪，也不会让STOP后永远停在识别姿态。
        // ----------------------------------------------------

        if (Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Fault)
        {
            Service_XY_Fault();
            return;
        }
        else if (Stop_Waiting_Safe_Disable)
        {
            if (Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Finish)
            {
                Holding_Jaw.Request_Return_To_Disable();
            }
            else if (Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Disable)
            {
                Pick_Action_Active = false;
                Stop_Waiting_Safe_Disable = false;
            }
        }
        else if (Holding_Jaw.Get_Grap_Failed())
        {
            Pick_Action_Active = false;
            // TOF超时等失败路径会安全回Disable，不得误报RECOGNITION_READY。
            Sorting_Handshake.Notify_Error(
                Grap_Error_To_Sorting_Error_Code(
                    Holding_Jaw.Get_Grap_Error()));
        }
        else
        {
            // ------------------------------------------------
            // 正常流程：
            //
            // 物体已经：
            // 抓取
            // -> 放到识别托盘
            // -> Lift抬起
            // -> 两根步进轴 + 三个舵机移动到专用Recognition姿态
            // -> 保持Recognition姿态
            //
            // 此时才通知Jetson打开侧面相机。
            // ------------------------------------------------

            if (
                Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Finish
                &&
                Sorting_Handshake
                    .Get_State()
                ==
                Sorting_Handshake_State_PICK_EXECUTING
                &&
                Sorting_Handshake
                    .Get_Running()
            )
            {
                // 通知只有在可靠TX队列接受后才推进握手状态。
                // 若队列暂时满，保留Pick_Action_Active，下个10ms仅重试通知。
                if (Sorting_Handshake.Notify_Recognition_Ready())
                {
                    Pick_Action_Active = false;
                }
            }
        }
    }


    // ========================================================
    // 识别托盘空载恢复：Recognition -> Disable
    //
    // 此路径只在第一阶段已经发送 RECOGNITION_READY、但 Jetson 连续确认
    // 识别托盘为空时触发。它不是 STOP：
    // - 不结束本场；
    // - 不改变成功计数/储物；
    // - 只把夹爪安全回 Disable；
    // - 到位后重新允许下一条 PICK_TARGET。
    // ========================================================

    if (!Retry_Pick_Return_Active && Sorting_Handshake.Pop_Retry_Pick_Request())
    {
        Retry_Pick_Return_Active = true;
        Holding_Jaw.Request_Return_To_Disable();
    }

    if (Retry_Pick_Return_Active)
    {
        if (Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Fault)
        {
            Service_XY_Fault();
            return;
        }
        else if (!Holding_Jaw.Get_Disable_Ready())
        {
            // UART/舵机短暂忙时允许下一个10ms继续请求，不把恢复事件丢掉。
            if (Holding_Jaw.Request_Return_To_Disable() == HAL_ERROR)
            {
                Sorting_Handshake.Notify_Error(Grap_Error_To_Sorting_Error_Code(Holding_Jaw.Get_Grap_Error()));
                Handle_Stop_Request();
                return;
            }
        }
        else if (Sorting_Handshake.Notify_Retry_Pick_Ready())
        {
            Retry_Pick_Return_Active = false;
        }
    }


    // ========================================================
    // 没有START时：
    //
    // 不接受/执行新的自动分拣任务。
    // ========================================================

    if (
        !Sorting_Handshake
            .Get_Running()
    )
    {
        return;
    }


    // ========================================================
    // 4.
    // 获取顶部相机识别得到的X/Y
    // ========================================================

    if (
        !Pick_Start_Pending
        &&
        !Pick_Action_Active
    )
    {
        Struct_Sorting_Pick_Request
            pick_request{};


        if (
            Sorting_Handshake
                .Pop_Pick_Request(
                    pick_request
                )
        )
        {
            Pending_Pick_Request =
                pick_request;


            Pick_Start_Pending =
                true;
        }
    }


    // ========================================================
    // 5.
    // 启动现有Holding Jaw抓取状态机
    //
    // Start_Grap可能暂时返回HAL_BUSY，
    // 例如Disable姿态UART还没完全发送完。
    //
    // 所以任务不能直接丢弃，要继续重试。
    // ========================================================

    if (Pick_Start_Pending)
    {
        ++Pick_Start_Wait_Count;
        if (Pick_Start_Wait_Count >= Pick_Start_Timeout_10ms)
        {
            Pick_Start_Pending = false;
            Pick_Start_Wait_Count = 0;
            Pending_Pick_Request = Struct_Sorting_Pick_Request{};
            Sorting_Handshake.Notify_Error(
                Sorting_Vision_Error_Grap_Start_Timeout);
            return;
        }

        const HAL_StatusTypeDef status =
            Holding_Jaw
                .Start_Grap(
                    Pending_Pick_Request.X_mm,

                    Pending_Pick_Request.Y_mm,

                    Convert_Grasp_Axis_To_Servo_Yaw_Rad(
                        Pending_Pick_Request.Grasp_Axis_Deg
                    ),

                    Pending_Pick_Request.Preopen_mm,

                    Pending_Pick_Request.Preopen_Valid
                );


        if (status == HAL_OK)
        {
            Pick_Start_Pending =
                false;

            Pick_Start_Wait_Count = 0;


            Pick_Action_Active =
                true;
        }
        else if (
            status ==
            HAL_ERROR
        )
        {
            // ------------------------------------------------
            // 例如：
            // TOF无效
            // XY位置无效
            // ------------------------------------------------

            Pick_Start_Pending =
                false;

            Pick_Start_Wait_Count = 0;


            Pending_Pick_Request =
                Struct_Sorting_Pick_Request{};


            Sorting_Handshake
                .Notify_Error(
                    Grap_Error_To_Sorting_Error_Code(
                        Holding_Jaw.Get_Grap_Error())
                );
        }

        // HAL_BUSY:
        //
        // 什么都不做，
        // 下一次10ms继续尝试。
    }


    // ========================================================
    // 6. 获取侧面识别结果并可靠启动储物机构
    // ========================================================
    if (!Sort_Action_Active && !Sort_Start_Pending)
    {
        Struct_Sorting_Result_Request sort_request{};
        if (Sorting_Handshake.Pop_Sort_Request(sort_request))
        {
            Pending_Sort_Request = sort_request;
            Sort_Target_Box_ID = sort_request.Box_ID;
            Sort_Start_Pending = true;
            Sort_Start_Wait_Count = 0;

            // ------------------------------------------------
            // 侧面识别已经完成，本件 SORT_RESULT 已收到。
            // 此时不再需要保持 Recognition 姿态，立即请求夹爪机构
            // Return -> Disable，为下一件顶部识别腾空视野。
            //
            // 储物机构可以同时执行本件投放；真正的 SORT_DONE
            // 会在“储物投放完成 + 夹爪已到 Disable”两个条件都满足后发送。
            // ------------------------------------------------
            Holding_Jaw.Request_Return_To_Disable();
        }
    }

    if (Sort_Start_Pending)
    {
        ++Sort_Start_Wait_Count;
        const HAL_StatusTypeDef status = Sorting_Storage.Start_Sort(
            Pending_Sort_Request.Box_ID);

        if (status == HAL_OK)
        {
            Sort_Start_Pending = false;
            Sort_Start_Wait_Count = 0;
            Sort_Action_Active = true;
        }
        else if (status == HAL_ERROR ||
                 Sort_Start_Wait_Count >= Sort_Start_Timeout_10ms)
        {
            Sort_Start_Pending = false;
            Sort_Start_Wait_Count = 0;
            Pending_Sort_Request = Struct_Sorting_Result_Request{};
            Sort_Target_Box_ID = 0;
            Sorting_Handshake.Notify_Error(2U);
        }
        // HAL_BUSY: 允许短暂重试，避免DM/UART瞬态导致已ACK任务直接丢失。
    }


    // ============================================================
    // 正式自动分拣：检查4310 + 翻转托盘完整投放流程是否完成
    //
    // STORAGE_TEST 的完成处理已经放到 Get_Running() 早退之前。
    // 这里仅处理正式 SORT_RESULT 链路。
    // ============================================================

    if (
        Sort_Action_Active
        &&
        Sorting_Storage
            .Pop_Sort_Finished()
    )
    {
        // 机械只完成一次；通知若暂时不能入TX队列，只重试通知，
        // 绝不重新执行机械动作。
        Sort_Action_Active = false;
        Sort_Done_Notify_Pending = true;
    }

    if (Sort_Done_Notify_Pending)
    {
        // ----------------------------------------------------
        // 下一件顶部视觉开始前，夹爪必须已经完全回到 Disable。
        // 如果仍在 Recognition/Return 过程中，只继续等待/重试回位，
        // 暂不发送 SORT_DONE，避免上位机开始下一件识别时夹爪挡住画面。
        // ----------------------------------------------------
        if (!Holding_Jaw.Get_Disable_Ready())
        {
            if (Holding_Jaw.Request_Return_To_Disable() == HAL_ERROR)
            {
                Sorting_Handshake.Notify_Error(Grap_Error_To_Sorting_Error_Code(Holding_Jaw.Get_Grap_Error()));
                Handle_Stop_Request();
                return;
            }
        }
        else if (Sorting_Storage.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
                 Sorting_Handshake.Notify_Sort_Done())
        {
            Sort_Done_Notify_Pending = false;
            Sort_Target_Box_ID = 0;
            Pending_Sort_Request = Struct_Sorting_Result_Request{};
        }
    }


    // ========================================================
    // 7.
    // 纯通信模拟模式
    //
    // 默认false。
    //
    // 如果你暂时想测试完整上下位机循环：
    //
    // 把头文件中的：
    //
    // Sort_Action_Simulation_Enable = false
    //
    // 改成：
    //
    // true
    //
    // 那么收到Box_ID一秒后自动发送SORT_DONE。
    // ========================================================

    if (
        Sort_Action_Active
        &&
        Sort_Action_Simulation_Enable
    )
    {
        Sort_Action_Simulation_Count++;


        if (
            Sort_Action_Simulation_Count >=
            Sort_Action_Simulation_Time_10ms
        )
        {
            Notify_Sort_Mechanism_Done();
        }
    }
}


// ============================================================
// 2ms
// ============================================================

void Class_Robot::
TIM_2ms_Calculate_PeriodElapsedCallback()
{
    Sorting_Storage
        .TIM_2ms_PeriodElapsedCallback();
}


// ============================================================
// 1ms
// ============================================================

void Class_Robot::
TIM_1ms_Calculate_Callback()
{
    // 推进Jetson UART DMA发送队列。10字节@115200bps约1ms内可完成，
    // 1ms服务一次可让同周期ACK/READY等连续帧可靠排出。
    Sorting_Vision_UART.Service_Tx();

    Holding_Jaw
        .TIM_1ms_PeriodElapsedCallback();
    if (Yaw_Stop_Pending && Holding_Jaw.Stop_Calibration_Yaw() == HAL_OK)
        Yaw_Stop_Pending = false;
}

// ============================================================
// 单个 180° 周期的闭合轴方向转换。
// 上下位机使用相同规则：先把相对标定参考轴的差角归一到 [-90,90)，
// 再换算舵机命令。TCP 补偿也必须使用这一个物理姿态。
// ============================================================
float Class_Robot::Convert_Grasp_Axis_To_Servo_Yaw_Rad(float Grasp_Axis_Robot_Deg) const
{
    if (!std::isfinite(Grasp_Axis_Robot_Deg) ||
        !std::isfinite(Pick_Yaw_Reference_Closing_Axis_Robot_Deg) ||
        !std::isfinite(Pick_Yaw_Reference_Servo_Command_Deg) ||
        !std::isfinite(Pick_Yaw_Servo_Deg_Per_Gripper_Deg) ||
        std::fabs(Pick_Yaw_Servo_Deg_Per_Gripper_Deg) < 1.0e-6f)
    {
        return NAN;
    }

    float delta_deg = std::fmod(
        Grasp_Axis_Robot_Deg - Pick_Yaw_Reference_Closing_Axis_Robot_Deg + 90.0f,
        180.0f);
    if (delta_deg < 0.0f) delta_deg += 180.0f;
    delta_deg -= 90.0f;

    const float servo_command_deg = Pick_Yaw_Reference_Servo_Command_Deg +
        Pick_Yaw_Servo_Deg_Per_Gripper_Deg * delta_deg;
    if (!std::isfinite(servo_command_deg) ||
        servo_command_deg < Pick_Yaw_Servo_Command_Min_Deg ||
        servo_command_deg > Pick_Yaw_Servo_Command_Max_Deg)
    {
        return NAN;
    }
    return servo_command_deg * PI / 180.0f;
}

HAL_StatusTypeDef Class_Robot::Confirm_XY_Origin_mm(float X_mm, float Y_mm)
{
    const uint32_t irq_state = __get_PRIMASK();
    __disable_irq();
    const auto jaw_state = Holding_Jaw.Get_Grap_Move_Status();
    const bool idle = !Sorting_Handshake.Get_Running() &&
        !Sorting_Start_Request_Pending &&
        !Pick_Action_Active && !Pick_Start_Pending && !Stop_Waiting_Safe_Disable &&
        !Retry_Pick_Return_Active && !Sort_Action_Active && !Sort_Start_Pending &&
        !Sort_Done_Notify_Pending && !Calib_Action_Active && !Yaw_Calib_Action_Active &&
        !Storage_Test_Start_Pending && !Storage_Test_Action_Active &&
        !Storage_Test_Done_Notify_Pending &&
        !Yaw_Stop_Pending && Sorting_Storage.Get_Disable_Ready() && !Sorting_Storage.Get_Fault() &&
        (jaw_state == Enum_Grap_Move_Disable || jaw_state == Enum_Grap_Move_Fault);
    HAL_StatusTypeDef status = HAL_BUSY;
    if (!std::isfinite(X_mm) || !std::isfinite(Y_mm))
    {
        status = HAL_ERROR;
    }
    else if (idle)
    {
        status = Holding_Jaw.Confirm_XY_Origin_mm(X_mm, Y_mm);
        if (status == HAL_OK)
        {
            XY_Fault_Reported = false;
        }
    }
    __set_PRIMASK(irq_state);
    return status;
}

HAL_StatusTypeDef Class_Robot::Start_XY_Collision_Homing()
{
    const bool safe = Sorting_Handshake.Get_State() == Sorting_Handshake_State_STOPPED &&
        !Sorting_Handshake.Get_Stop_Ack_Pending() && !Get_Mechanism_Action_Active() &&
        Holding_Jaw.Get_Homing_Ready() && Sorting_Storage.Get_Disable_Ready() &&
        !Sorting_Storage.Get_Fault();
    if (!safe) return HAL_BUSY;
    return Holding_Jaw.Start_XY_Collision_Homing();
}

void Class_Robot::Stop_XY_Collision_Homing()
{
    Holding_Jaw.Stop_XY_Collision_Homing();
}

bool Class_Robot::Service_XY_Fault()
{
    const bool xy_in_use = Pick_Action_Active || Pick_Start_Pending ||
        Calib_Action_Active || Stop_Waiting_Safe_Disable || Retry_Pick_Return_Active ||
        Sort_Done_Notify_Pending ||
        Holding_Jaw.Get_Grap_Move_Status() != Enum_Grap_Move_Disable;
    const bool xy_bad = Holding_Jaw.XY_Motion.Get_Fault() ||
        (xy_in_use && !Holding_Jaw.XY_Motion.Get_Position_Valid()) ||
        Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Fault;
    if (!xy_bad)
    {
        return false;
    }

    Holding_Jaw.Abort_XY_Fault();
    Sorting_Start_Request_Pending = false;
    Pick_Start_Pending = false;
    Pick_Action_Active = false;
    Pick_Start_Wait_Count = 0U;
    Pending_Pick_Request = Struct_Sorting_Pick_Request{};
    Stop_Waiting_Safe_Disable = false;
    Retry_Pick_Return_Active = false;
    Calib_Action_Active = false;
    Calib_Start_Pending = false;
    Calib_Move_Count = 0U;
    Calib_Stable_Count = 0U;
    if (Yaw_Calib_Action_Active)
    {
        Yaw_Stop_Pending = true;
        Yaw_Calib_Action_Active = false;
        Yaw_Calib_Start_Pending = false;
        Yaw_Calib_Count_10ms = 0U;
    }
    Sort_Action_Active = false;
    Sort_Start_Pending = false;
    Sort_Done_Notify_Pending = false;
    Sort_Start_Wait_Count = 0U;
    Sort_Target_Box_ID = 0U;
    Pending_Sort_Request = Struct_Sorting_Result_Request{};
    Storage_Test_Start_Pending = false;
    Storage_Test_Action_Active = false;
    Storage_Test_Done_Notify_Pending = false;
    Storage_Test_Box_ID = 0U;
    Storage_Test_Start_Wait_Count = 0U;
    if (!XY_Fault_Reported)
    {
        XY_Fault_Reported = true;
        // 只请求一次：每 10ms 重复 Request_Disable 会重置储物回程状态，
        // 使安全停机无法完成。
        if (!Sorting_Storage.Get_Fault()) Sorting_Storage.Request_Disable();
        // Existing Jetson protocol retains its XY error code; detailed drive
        // 更详细的原因可通过 MCU 的 XY_Motion 诊断数据读取。
        Sorting_Handshake.Notify_Error(Grap_Error_To_Sorting_Error_Code(Holding_Jaw.Get_Grap_Error()));
    }
    return true;
}
