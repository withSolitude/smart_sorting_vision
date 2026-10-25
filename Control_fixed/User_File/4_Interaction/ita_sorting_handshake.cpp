#include "4_Interaction/ita_sorting_handshake.h"
#include "3_Robot/claw_preopen_profile.h"
#include "3_Robot/xy_motion_config.h"

#include <cmath>


namespace
{
    constexpr float
    SORTING_PICK_DUPLICATE_EPS_MM =
        0.051f;
}


// ============================================================
// Init
// ============================================================

void Class_Sorting_Handshake::Init(
    Class_Sorting_Vision_UART *__Communication)
{
    Communication =
        __Communication;
    Explicit_Stop_Request_Pending = false;
    Error_Notify_Pending = false;
    Pending_Error_Code = 0U;


    State =
        Sorting_Handshake_State_STOPPED;


    Current_Pick =
        Struct_Sorting_Pick_Request{};
    Pending_Preopen_mm = 0.0f;
    Pending_Preopen_Valid = false;
    Pending_Yaw_Branch = 0U;


    Current_Sort =
        Struct_Sorting_Result_Request{};


    Current_Calib =
        Struct_Sorting_Calib_Request{};

    Current_Yaw_Calib =
        Struct_Sorting_Yaw_Calib_Request{};


    Pick_Request_Pending =
        false;


    Sort_Request_Pending =
        false;

    Retry_Pick_Request_Pending =
        false;

    Storage_Test_Request_Pending =
        false;

    Storage_Test_In_Progress =
        false;

    Storage_Test_Box_ID =
        0U;


    Calib_Request_Pending =
        false;

    Yaw_Calib_Request_Pending =
        false;


    Stop_Request_Pending =
        false;

    Stop_Ack_Pending =
        false;


    Last_Completed_Box =
        0U;


    Last_Sort_Done_Valid =
        false;
}


// ============================================================
// Process
// ============================================================

void Class_Sorting_Handshake::
Process()
{
    if (Communication == nullptr)
    {
        return;
    }


    Struct_Sorting_Vision_UART_Data
        data{};

    if (Error_Notify_Pending && Communication->Send_Error(Pending_Error_Code) == HAL_OK)
        Error_Notify_Pending = false;


    while (
        Communication->
            Pop_Command(data)
    )
    {
        Handle_Frame(
            data
        );
    }
}


// ============================================================
// STM32实体按键START
// ============================================================

bool Class_Sorting_Handshake::
Request_Start_Work()
{
    if (Communication == nullptr)
    {
        return false;
    }


    if (
        State !=
        Sorting_Handshake_State_STOPPED
        ||
        Stop_Ack_Pending
        ||
        Storage_Test_In_Progress
        ||
        Error_Notify_Pending
    )
    {
        // Jetson主动STOP的安全ACK尚未完成时，不能抢先START。
        return false;
    }


    // ========================================================
    // 先通知Jetson
    // ========================================================

    const HAL_StatusTypeDef status =
        Communication->
            Send_Start_Work();


    if (status != HAL_OK)
    {
        return false;
    }

    // 这里已经完成一次新的START握手。清除上一次掉线安全STOP留下的
    // “显式停止”标志，避免后续普通机械故障被误判成操作员STOP。
    Explicit_Stop_Request_Pending = false;


    // ========================================================
    // 清空上一轮
    // ========================================================

    Current_Pick =
        Struct_Sorting_Pick_Request{};
    Pending_Preopen_mm = 0.0f;
    Pending_Preopen_Valid = false;
    Pending_Yaw_Branch = 0U;


    Current_Sort =
        Struct_Sorting_Result_Request{};

    Current_Calib =
        Struct_Sorting_Calib_Request{};

    Current_Yaw_Calib =
        Struct_Sorting_Yaw_Calib_Request{};


    Pick_Request_Pending =
        false;


    Sort_Request_Pending =
        false;

    Retry_Pick_Request_Pending =
        false;

    Storage_Test_Request_Pending =
        false;

    Storage_Test_In_Progress =
        false;

    Storage_Test_Box_ID =
        0U;


    Calib_Request_Pending =
        false;

    Yaw_Calib_Request_Pending =
        false;


    Stop_Request_Pending =
        false;

    Stop_Ack_Pending =
        false;


    Last_Completed_Box =
        0U;


    Last_Sort_Done_Valid =
        false;


    // ========================================================
    // 从现在开始等待顶部相机X/Y
    // ========================================================

    State =
        Sorting_Handshake_State_WAIT_PICK_TARGET;


    return true;
}


// ============================================================
// STM32实体按键STOP
// ============================================================

bool Class_Sorting_Handshake::
Request_Stop_Work()
{
    Explicit_Stop_Request_Pending = true;
    // ========================================================
    // 不管UART是否正常，本机都必须停
    // ========================================================

    Pick_Request_Pending =
        false;


    Sort_Request_Pending =
        false;

    Retry_Pick_Request_Pending =
        false;

    Storage_Test_Request_Pending =
        false;

    Storage_Test_In_Progress =
        false;

    Storage_Test_Box_ID =
        0U;


    Calib_Request_Pending =
        false;

    Yaw_Calib_Request_Pending =
        false;


    Stop_Request_Pending =
        true;


    Current_Pick =
        Struct_Sorting_Pick_Request{};
    Pending_Preopen_mm = 0.0f;
    Pending_Preopen_Valid = false;
    Pending_Yaw_Branch = 0U;


    Current_Sort =
        Struct_Sorting_Result_Request{};


    Current_Calib =
        Struct_Sorting_Calib_Request{};

    Current_Yaw_Calib =
        Struct_Sorting_Yaw_Calib_Request{};


    Last_Sort_Done_Valid =
        false;


    State =
        Sorting_Handshake_State_STOPPED;


    // ========================================================
    // 通知Jetson停止相机和自动识别
    // ========================================================

    if (Communication == nullptr)
    {
        return false;
    }

    // STOP建立后，丢弃此前尚未消费的普通命令，禁止同批旧命令重新激活动作。
    Communication->Clear_Command_Queue();

    return
        Communication->
            Send_Stop_Work()
        ==
        HAL_OK;
}


// ============================================================
// Handle frame
// ============================================================

void Class_Sorting_Handshake::
Handle_Frame(
    const Struct_Sorting_Vision_UART_Data &Data)
{
    if (Communication == nullptr)
    {
        return;
    }


    switch (Data.Mode)
    {
        // ====================================================
        // HEARTBEAT
        // ====================================================

        case Sorting_Vision_Mode_HEARTBEAT:
        {
            Communication->
                Send_Heartbeat_Ack();

            break;
        }


        // ====================================================
        // 动态夹爪预开口参数
        // ====================================================

        case Sorting_Vision_Mode_PICK_PREOPEN:
        {
            if (State == Sorting_Handshake_State_WAIT_PICK_TARGET)
            {
                // 在确认前先检查请求开口是否在当前线性公式的安全范围内。
                // 错误 8：预开口公式未配置、参数非法或请求超出范围。
                float unused_angle_deg = 0.0f;
                if (std::fabs(Data.Y_mm) > 0.051f ||
                    !ClawPreopen::ToAngle(Data.X_mm, unused_angle_deg))
                {
                    Pending_Preopen_Valid = false;
                    Pending_Preopen_mm = 0.0f;
                    Notify_Error(8U);
                    break;
                }

                if (Communication->Send_Pick_Preopen_Ack(Data.X_mm) != HAL_OK)
                {
                    break;
                }

                Pending_Preopen_mm = Data.X_mm;
                Pending_Preopen_Valid = true;
                break;
            }

            Communication->Send_Busy();
            break;
        }


        // ====================================================
        // 双边Yaw物理分支参数
        // ====================================================

        case Sorting_Vision_Mode_PICK_YAW_BRANCH:
        {
            if (State == Sorting_Handshake_State_WAIT_PICK_TARGET)
            {
                // UART层已经限制Yaw_Branch只能为0/1。
                // ACK可靠进入发送队列后再锁存，避免上位机认为参数已接受但MCU未保存。
                if (Communication->Send_Pick_Yaw_Branch_Ack(Data.Yaw_Branch) != HAL_OK)
                {
                    break;
                }

                Pending_Yaw_Branch = Data.Yaw_Branch;
                break;
            }

            Communication->Send_Busy();
            break;
        }


        // ====================================================
        // 顶部相机X/Y
        // ====================================================

        case Sorting_Vision_Mode_PICK_TARGET:
        {
            // =================================================
            // 只有已经按START以后才接受X/Y
            // =================================================

            if (
                State ==
                Sorting_Handshake_State_WAIT_PICK_TARGET
            )
            {
                // 先确保ACK已进入可靠发送队列，再把机械任务置为可执行。
                // 如果TX队列异常满，不允许“动作已经开始但上位机从未收到ACK”。
                if (Communication->Send_Pick_Target_Ack(
                        Data.X_mm, Data.Y_mm, Data.Grasp_Axis_Deg) != HAL_OK)
                {
                    break;
                }

                Current_Pick.X_mm = Data.X_mm;
                Current_Pick.Y_mm = Data.Y_mm;
                Current_Pick.Grasp_Axis_Deg = Data.Grasp_Axis_Deg;
                Current_Pick.Yaw_Branch = Pending_Yaw_Branch;
                Current_Pick.Preopen_mm = Pending_Preopen_mm;
                Current_Pick.Preopen_Valid = Pending_Preopen_Valid;
                Current_Pick.Sequence = Data.Sequence;

                // 参数只属于当前这一件，锁定后立刻清空“下一件”缓存。
                Pending_Preopen_mm = 0.0f;
                Pending_Preopen_Valid = false;
                Pending_Yaw_Branch = 0U;

                Pick_Request_Pending = true;
                Sort_Request_Pending = false;

                // 一旦真正开始下一件，上一件SORT_DONE重复保护失效。
                Last_Sort_Done_Valid = false;
                State = Sorting_Handshake_State_PICK_EXECUTING;
                break;
            }


            // =================================================
            // PICK ACK丢失：
            //
            // PC重发同样的X/Y时只ACK，
            // 不允许再次启动抓取。
            // =================================================

            if (
                (
                    State ==
                    Sorting_Handshake_State_PICK_EXECUTING
                    ||
                    State ==
                    Sorting_Handshake_State_WAIT_SORT_RESULT
                )
                &&
                Is_Same_Pick(Data)
            )
            {
                Communication->
                    Send_Pick_Target_Ack(
                        Current_Pick.X_mm,
                        Current_Pick.Y_mm,
                        Current_Pick.Grasp_Axis_Deg
                    );


                // READY可能丢失。
                //
                // 如果物体已经到识别台，
                // 重发READY。
                if (
                    State ==
                    Sorting_Handshake_State_WAIT_SORT_RESULT
                )
                {
                    Communication->
                        Send_Recognition_Ready(
                            Current_Pick.X_mm,
                            Current_Pick.Y_mm
                        );
                }


                break;
            }


            Communication->
                Send_Busy();


            break;
        }


        // ====================================================
        // 识别托盘空载：放弃本次失败夹取并安全回位
        // ====================================================

        case Sorting_Vision_Mode_RETRY_PICK:
        {
            if (State == Sorting_Handshake_State_WAIT_SORT_RESULT)
            {
                // 不发送“立即ACK”。上位机需要等机械真正回到 Disable
                // 才能重新打开顶部识别，因此 READY 由机械层完成后发送。
                Retry_Pick_Request_Pending = true;
                Sort_Request_Pending = false;
                Last_Sort_Done_Valid = false;
                State = Sorting_Handshake_State_RETRY_RETURNING;
                break;
            }

            if (State == Sorting_Handshake_State_RETRY_RETURNING)
            {
                // 上位机按 retry_interval 重发是正常的。恢复尚未完成时
                // 不重复触发机械动作，等待完成后统一回 READY。
                break;
            }

            if (State == Sorting_Handshake_State_WAIT_PICK_TARGET)
            {
                // READY 可能丢失。此时机械已经安全回位，重复 RETRY_PICK
                // 只补发 READY，不改变任何机械状态。
                Communication->Send_Retry_Pick_Ready();
                break;
            }

            Communication->Send_Busy();
            break;
        }


        // ====================================================
        // 侧面识别Box ID
        // ====================================================

        case Sorting_Vision_Mode_SORT_RESULT:
        {
            if (
                State ==
                Sorting_Handshake_State_WAIT_SORT_RESULT
            )
            {
                // UART层已经检查Box_ID 1~6。先把ACK排入发送队列，再执行投放。
                if (Communication->Send_Sort_Result_Ack(Data.Box_ID) != HAL_OK)
                {
                    break;
                }

                Current_Sort.Box_ID = Data.Box_ID;
                Current_Sort.Sequence = Data.Sequence;
                Sort_Request_Pending = true;
                State = Sorting_Handshake_State_SORT_EXECUTING;
                break;
            }


            // =================================================
            // SORT_RESULT_ACK丢失
            // =================================================

            if (
                State ==
                Sorting_Handshake_State_SORT_EXECUTING
                &&
                Data.Box_ID ==
                Current_Sort.Box_ID
            )
            {
                Communication->
                    Send_Sort_Result_Ack(
                        Current_Sort.Box_ID
                    );


                break;
            }


            // =================================================
            // SORT_DONE丢失
            //
            // STM32已经完成上一件，
            // Jetson重发上一个SORT_RESULT。
            // =================================================

            if (
                State ==
                Sorting_Handshake_State_WAIT_PICK_TARGET
                &&
                Last_Sort_Done_Valid
                &&
                Data.Box_ID ==
                Last_Completed_Box
            )
            {
                Communication->
                    Send_Sort_Done(
                        Last_Completed_Box
                    );


                break;
            }


            Communication->
                Send_Busy();


            break;
        }


        // ====================================================
        // 储物机构独立测试
        //
        // 只允许 STOPPED 状态使用，避免测试命令与正式自动分拣并发。
        // 这里只排队测试请求，不改变 State。
        // 真正机械动作由 Class_Robot 消费。
        // ====================================================
        case Sorting_Vision_Mode_STORAGE_TEST:
        {
            if (State != Sorting_Handshake_State_STOPPED || Stop_Ack_Pending)
            {
                Communication->Send_Busy();
                break;
            }

            if (Data.Box_ID < 1U || Data.Box_ID > 6U)
            {
                Communication->Send_Error(2U);
                break;
            }

            // 同一测试命令在ACK丢失时允许重复发送ACK，
            // 但从“已排队”一直到“机械DONE”之间都不重复启动机械。
            if (Storage_Test_In_Progress)
            {
                if (Storage_Test_Box_ID == Data.Box_ID)
                {
                    Communication->Send_Storage_Test_Ack(Storage_Test_Box_ID);
                }
                else
                {
                    Communication->Send_Busy();
                }
                break;
            }

            if (Communication->Send_Storage_Test_Ack(Data.Box_ID) != HAL_OK)
            {
                break;
            }

            Storage_Test_Box_ID = Data.Box_ID;
            Storage_Test_Request_Pending = true;
            Storage_Test_In_Progress = true;
            break;
        }


        // ====================================================
        // 独立XY标定：仅允许STOPPED状态进入
        // ====================================================
        case Sorting_Vision_Mode_CALIB_MOVE:
        {
            if (State == Sorting_Handshake_State_STOPPED && !Stop_Ack_Pending && !Storage_Test_In_Progress)
            {
                if (!std::isfinite(Data.X_mm) || !std::isfinite(Data.Y_mm) ||
                    Data.X_mm < XY_Config::X_Min_mm || Data.X_mm > XY_Config::X_Max_mm ||
                    Data.Y_mm < XY_Config::Y_Min_mm || Data.Y_mm > XY_Config::Y_Max_mm)
                {
                    Communication->Send_Error(5U);
                    break;
                }

                if (Communication->Send_Calib_Move_Ack(Data.X_mm, Data.Y_mm) != HAL_OK)
                {
                    break;
                }

                Current_Calib.X_mm = Data.X_mm;
                Current_Calib.Y_mm = Data.Y_mm;
                Current_Calib.Sequence = Data.Sequence;
                Calib_Request_Pending = true;
                State = Sorting_Handshake_State_CALIB_EXECUTING;
                break;
            }

            if (State == Sorting_Handshake_State_CALIB_EXECUTING &&
                std::fabs(Data.X_mm - Current_Calib.X_mm) <= SORTING_PICK_DUPLICATE_EPS_MM &&
                std::fabs(Data.Y_mm - Current_Calib.Y_mm) <= SORTING_PICK_DUPLICATE_EPS_MM)
            {
                Communication->Send_Calib_Move_Ack(Current_Calib.X_mm, Current_Calib.Y_mm);
                break;
            }

            Communication->Send_Busy();
            break;
        }

        // ====================================================
        // 独立Yaw标定：仅允许STOPPED状态进入。
        //
        // CALIB_YAW复用协议X字段承载“直接舵机命令角度(deg)”。
        // 它不会执行XY、下降、夹取或储物动作。
        // ====================================================
        case Sorting_Vision_Mode_CALIB_YAW:
        {
            constexpr float SERVO_MIN_DEG = -135.0f;
            constexpr float SERVO_MAX_DEG =  135.0f;

            if (State == Sorting_Handshake_State_STOPPED && !Stop_Ack_Pending && !Storage_Test_In_Progress)
            {
                const float servo_deg = Data.X_mm;
                if (!std::isfinite(servo_deg) ||
                    servo_deg < SERVO_MIN_DEG ||
                    servo_deg > SERVO_MAX_DEG ||
                    std::fabs(Data.Y_mm) > 0.051f)
                {
                    Communication->Send_Error(6U);
                    break;
                }

                if (Communication->Send_Calib_Yaw_Ack(servo_deg) != HAL_OK)
                {
                    break;
                }

                Current_Yaw_Calib.Servo_Command_Deg = servo_deg;
                Current_Yaw_Calib.Sequence = Data.Sequence;
                Yaw_Calib_Request_Pending = true;
                State = Sorting_Handshake_State_CALIB_YAW_EXECUTING;
                break;
            }

            if (State == Sorting_Handshake_State_CALIB_YAW_EXECUTING &&
                std::fabs(Data.X_mm - Current_Yaw_Calib.Servo_Command_Deg) <= 0.051f)
            {
                Communication->Send_Calib_Yaw_Ack(
                    Current_Yaw_Calib.Servo_Command_Deg);
                break;
            }

            Communication->Send_Busy();
            break;
        }

        // ====================================================
        // Jetson主动STOP
        //
        // 虽然正常使用实体按键STOP，
        // 仍保留PC急停能力。
        // ====================================================

        case Sorting_Vision_Mode_STOP:
        {
            Explicit_Stop_Request_Pending = true;
            Retry_Pick_Request_Pending = false;
            Storage_Test_Request_Pending = false;
            Storage_Test_In_Progress = false;
            Storage_Test_Box_ID = 0U;
            Pending_Preopen_mm = 0.0f;
            Pending_Preopen_Valid = false;
            Pick_Request_Pending =
                false;


            Sort_Request_Pending =
                false;

            Calib_Request_Pending =
                false;

            Yaw_Calib_Request_Pending =
                false;


            Stop_Request_Pending =
                true;

            // 不在这里立即ACK。
            // 机械层真正回到安全可START状态后由Notify_Stop_Ready()发送。
            Stop_Ack_Pending =
                true;


            Current_Pick =
                Struct_Sorting_Pick_Request{};
            Pending_Yaw_Branch = 0U;


            Current_Sort =
                Struct_Sorting_Result_Request{};

            Current_Calib =
                Struct_Sorting_Calib_Request{};

            Current_Yaw_Calib =
                Struct_Sorting_Yaw_Calib_Request{};


            Last_Sort_Done_Valid =
                false;


            State =
                Sorting_Handshake_State_STOPPED;

            // STOP后当前批次其余普通命令全部作废。
            Communication->Clear_Command_Queue();

            // MCU_STOP_ACK 延迟到机械层确认安全回位后再发送。
            break;
        }


        default:
        {
            // STM32状态帧正常不会由PC发来。
            break;
        }
    }
}


// ============================================================
// Pop Pick
// ============================================================

bool Class_Sorting_Handshake::
Pop_Pick_Request(
    Struct_Sorting_Pick_Request &Request_Out)
{
    if (!Pick_Request_Pending)
    {
        return false;
    }


    Request_Out =
        Current_Pick;


    Pick_Request_Pending =
        false;


    return true;
}


// ============================================================
// Pop Sort
// ============================================================

bool Class_Sorting_Handshake::
Pop_Sort_Request(
    Struct_Sorting_Result_Request &Request_Out)
{
    if (!Sort_Request_Pending)
    {
        return false;
    }


    Request_Out =
        Current_Sort;


    Sort_Request_Pending =
        false;


    return true;
}


// ============================================================
// 识别台空载恢复请求
// ============================================================

bool Class_Sorting_Handshake::
Pop_Retry_Pick_Request()
{
    if (!Retry_Pick_Request_Pending)
    {
        return false;
    }

    Retry_Pick_Request_Pending = false;
    return true;
}


bool Class_Sorting_Handshake::
Notify_Retry_Pick_Ready()
{
    if (
        Communication == nullptr
        ||
        State != Sorting_Handshake_State_RETRY_RETURNING
    )
    {
        return false;
    }

    const HAL_StatusTypeDef status =
        Communication->Send_Retry_Pick_Ready();

    if (status != HAL_OK)
    {
        return false;
    }

    // READY 已经可靠进入发送队列后才清空上一笔失败事务并开放新 PICK。
    Current_Pick = Struct_Sorting_Pick_Request{};
    Pending_Yaw_Branch = 0U;
    Current_Sort = Struct_Sorting_Result_Request{};
    Pick_Request_Pending = false;
    Sort_Request_Pending = false;
    Retry_Pick_Request_Pending = false;
    Last_Sort_Done_Valid = false;
    State = Sorting_Handshake_State_WAIT_PICK_TARGET;
    return true;
}


// ============================================================
// 第一阶段完成
// ============================================================

bool Class_Sorting_Handshake::
Notify_Recognition_Ready()
{
    if (
        Communication == nullptr
        ||
        State !=
        Sorting_Handshake_State_PICK_EXECUTING
    )
    {
        return false;
    }


    const HAL_StatusTypeDef status =
        Communication->Send_Recognition_Ready(
            Current_Pick.X_mm,
            Current_Pick.Y_mm);

    if (status != HAL_OK)
    {
        return false;
    }

    State = Sorting_Handshake_State_WAIT_SORT_RESULT;
    return true;
}


// ============================================================
// 第二阶段完成
// ============================================================

bool Class_Sorting_Handshake::
Notify_Sort_Done()
{
    if (
        Communication == nullptr
        ||
        State !=
        Sorting_Handshake_State_SORT_EXECUTING
    )
    {
        return false;
    }


    const uint8_t completed_box = Current_Sort.Box_ID;
    const HAL_StatusTypeDef status =
        Communication->Send_Sort_Done(completed_box);

    // 只有DONE已经可靠排队后才能推进状态并清空事务。
    if (status != HAL_OK)
    {
        return false;
    }

    Last_Completed_Box = completed_box;
    Last_Sort_Done_Valid = true;

    Current_Pick = Struct_Sorting_Pick_Request{};
    Pending_Yaw_Branch = 0U;
    Current_Sort = Struct_Sorting_Result_Request{};
    State = Sorting_Handshake_State_WAIT_PICK_TARGET;
    return true;
}


// ============================================================
// 取出 XY 标定请求
// ============================================================
bool Class_Sorting_Handshake::Pop_Storage_Test_Request(
    uint8_t &Box_ID_Out)
{
    if (!Storage_Test_Request_Pending)
    {
        return false;
    }

    Box_ID_Out = Storage_Test_Box_ID;
    Storage_Test_Request_Pending = false;
    return true;
}


bool Class_Sorting_Handshake::Notify_Storage_Test_Done(
    const uint8_t Box_ID)
{
    if (Communication == nullptr ||
        State != Sorting_Handshake_State_STOPPED ||
        !Storage_Test_In_Progress || Box_ID != Storage_Test_Box_ID ||
        Box_ID < 1U ||
        Box_ID > 6U)
    {
        return false;
    }

    const HAL_StatusTypeDef status =
        Communication->Send_Storage_Test_Done(Box_ID);

    if (status == HAL_OK)
    {
        Storage_Test_Request_Pending = false;
        Storage_Test_In_Progress = false;
        Storage_Test_Box_ID = 0U;
    }

    return status == HAL_OK;
}


// ============================================================
// XY calibration request
// ============================================================

bool Class_Sorting_Handshake::Pop_Calib_Request(
    Struct_Sorting_Calib_Request &Request_Out)
{
    if (!Calib_Request_Pending)
    {
        return false;
    }

    Request_Out = Current_Calib;
    Calib_Request_Pending = false;
    return true;
}


bool Class_Sorting_Handshake::Notify_Calib_Reached(
    const float X_mm, const float Y_mm)
{
    if (Communication == nullptr ||
        State != Sorting_Handshake_State_CALIB_EXECUTING)
    {
        return false;
    }

    const HAL_StatusTypeDef status =
        Communication->Send_Calib_Reached(X_mm, Y_mm);

    if (status != HAL_OK)
    {
        return false;
    }

    Current_Calib = Struct_Sorting_Calib_Request{};
    Calib_Request_Pending = false;
    State = Sorting_Handshake_State_STOPPED;
    return true;
}


// ============================================================
// 取出 Yaw 标定请求
// 关联修订（2026-09-11）：保持声明与类外定义一致。
// ============================================================

bool Class_Sorting_Handshake::Pop_Yaw_Calib_Request(
    Struct_Sorting_Yaw_Calib_Request &Request_Out)
{
    if (!Yaw_Calib_Request_Pending)
    {
        return false;
    }

    Request_Out = Current_Yaw_Calib;
    Yaw_Calib_Request_Pending = false;
    return true;
}


bool Class_Sorting_Handshake::Notify_Yaw_Calib_Reached(
    const float Servo_Command_Deg)
{
    if (Communication == nullptr ||
        State != Sorting_Handshake_State_CALIB_YAW_EXECUTING)
    {
        return false;
    }

    const HAL_StatusTypeDef status =
        Communication->Send_Calib_Yaw_Reached(Servo_Command_Deg);

    if (status != HAL_OK)
    {
        return false;
    }

    Current_Yaw_Calib = Struct_Sorting_Yaw_Calib_Request{};
    Yaw_Calib_Request_Pending = false;
    State = Sorting_Handshake_State_STOPPED;
    return true;
}


// ============================================================
// 取出停机请求
// ============================================================

bool Class_Sorting_Handshake::
Pop_Stop_Request(bool *Explicit_Recovery)
{
    if (!Stop_Request_Pending)
    {
        return false;
    }


    Stop_Request_Pending =
        false;
    if (Explicit_Recovery != nullptr) *Explicit_Recovery = Explicit_Stop_Request_Pending;
    Explicit_Stop_Request_Pending = false;


    return true;
}


// ============================================================
// Jetson主动STOP安全完成ACK
// ============================================================

bool Class_Sorting_Handshake::
Notify_Stop_Ready()
{
    if (
        !Stop_Ack_Pending
        ||
        Communication == nullptr
    )
    {
        return false;
    }

    // ERROR可以通过Jetson主动STOP进入安全恢复链。
    // 机械层只有在真实安全条件满足后才会调用这里，因此此刻把握手
    // 统一收敛到STOPPED，再发ACK作为“上下位机同步边界”。
    State =
        Sorting_Handshake_State_STOPPED;

    if (
        Communication->
            Send_Stop_Ack()
        !=
        HAL_OK
    )
    {
        return false;
    }

    Stop_Ack_Pending =
        false;

    return true;
}


// ============================================================
// 错误处理
// ============================================================

void Class_Sorting_Handshake::
Notify_Error(
    const uint8_t &Error_Code)
{
    Pick_Request_Pending =
        false;


    Sort_Request_Pending =
        false;

    Retry_Pick_Request_Pending =
        false;

    Storage_Test_Request_Pending =
        false;

    Storage_Test_In_Progress =
        false;

    Storage_Test_Box_ID =
        0U;


    Calib_Request_Pending =
        false;

    Yaw_Calib_Request_Pending =
        false;


    Current_Calib = Struct_Sorting_Calib_Request{};
    Current_Yaw_Calib = Struct_Sorting_Yaw_Calib_Request{};
    Pending_Preopen_mm = 0.0f;
    Pending_Preopen_Valid = false;
    Pending_Yaw_Branch = 0U;


    // 机械层进入安全返回。
    Stop_Request_Pending =
        true;


    // Preserve an operator STOP that arrived in the same scheduler cycle as
    // 本次故障不能覆盖同一调度周期内已经收到的显式恢复请求。
    State = Explicit_Stop_Request_Pending ? Sorting_Handshake_State_STOPPED : Sorting_Handshake_State_ERROR;


    Pending_Error_Code = Error_Code <= Sorting_Vision_Error_Grap_TOF_Timeout
        ? Error_Code : static_cast<uint8_t>(Sorting_Vision_Error_None);
    Error_Notify_Pending = true;
    if (Communication != nullptr && Communication->Send_Error(Pending_Error_Code) == HAL_OK)
        Error_Notify_Pending = false;
}


// ============================================================
// Reset stopped
// ============================================================

void Class_Sorting_Handshake::
Reset_To_Stopped()
{
    Explicit_Stop_Request_Pending = false;
    Current_Pick =
        Struct_Sorting_Pick_Request{};
    Pending_Yaw_Branch = 0U;


    Current_Sort =
        Struct_Sorting_Result_Request{};


    Pick_Request_Pending =
        false;


    Sort_Request_Pending =
        false;

    Retry_Pick_Request_Pending =
        false;

    Storage_Test_Request_Pending =
        false;

    Storage_Test_In_Progress =
        false;

    Storage_Test_Box_ID =
        0U;


    Calib_Request_Pending =
        false;

    Yaw_Calib_Request_Pending =
        false;


    Current_Calib = Struct_Sorting_Calib_Request{};
    Current_Yaw_Calib = Struct_Sorting_Yaw_Calib_Request{};


    Stop_Request_Pending =
        false;

    Stop_Ack_Pending =
        false;


    Last_Completed_Box =
        0U;


    Last_Sort_Done_Valid =
        false;


    State =
        Sorting_Handshake_State_STOPPED;
}


// ============================================================
// 判断重复PICK
// ============================================================

bool Class_Sorting_Handshake::
Is_Same_Pick(
    const Struct_Sorting_Vision_UART_Data &Data)
    const
{
    return
        std::fabs(
            Data.X_mm -
            Current_Pick.X_mm
        )
        <=
        SORTING_PICK_DUPLICATE_EPS_MM
        &&
        std::fabs(
            Data.Y_mm -
            Current_Pick.Y_mm
        )
        <=
        SORTING_PICK_DUPLICATE_EPS_MM
        &&
        std::fabs(
            std::remainder(
                Data.Grasp_Axis_Deg - Current_Pick.Grasp_Axis_Deg,
                180.0f
            )
        ) <= 1.01f;
}
