#ifndef ITA_SORTING_HANDSHAKE_H
#define ITA_SORTING_HANDSHAKE_H


#include <cstdint>

#include "2_Device/PC_Communication/dvc_sorting_vision_uart_v2.h"


// ============================================================
// 分拣握手状态
// ============================================================

enum Enum_Sorting_Handshake_State : uint8_t
{
    // ========================================================
    // 整机自动分拣没有启动
    //
    // 机构保持Disable待机姿态。
    // ========================================================

    Sorting_Handshake_State_STOPPED =
        0,


    // ========================================================
    // 已按下启动按键
    //
    // Jetson应该运行顶部相机，
    // STM32等待PICK_TARGET(X,Y)。
    // ========================================================

    Sorting_Handshake_State_WAIT_PICK_TARGET,


    // ========================================================
    // 已经收到X/Y
    //
    // 下位机正在：
    //
    // XY移动
    // -> 抓取
    // -> 搬运
    // -> 放到侧面识别托盘
    // -> 回Disable
    // ========================================================

    Sorting_Handshake_State_PICK_EXECUTING,


    // ========================================================
    // 第一阶段已经完成
    //
    // 等待Jetson侧面识别结果：
    //
    // SORT_RESULT(Box_ID)
    // ========================================================

    Sorting_Handshake_State_WAIT_SORT_RESULT,


    // ========================================================
    // 侧面识别台连续确认无物体。
    // 正在等待夹爪从 Recognition 姿态安全返回 Disable。
    // 完成后重新进入 WAIT_PICK_TARGET，不结束整场比赛。
    // ========================================================

    Sorting_Handshake_State_RETRY_RETURNING,


    // ========================================================
    // 已经收到Box_ID
    //
    // 后续：
    //
    // 4310转储物盒
    // -> 托盘翻转
    // -> 投放
    // -> 回Disable
    // ========================================================

    Sorting_Handshake_State_SORT_EXECUTING,

    // 独立XY标定运动，不进入自动分拣流程
    Sorting_Handshake_State_CALIB_EXECUTING,

    // 独立Yaw标定运动，不进入自动分拣流程
    Sorting_Handshake_State_CALIB_YAW_EXECUTING,


    // ========================================================
    // 错误
    // ========================================================

    Sorting_Handshake_State_ERROR,
};


// ============================================================
// 第一阶段任务
// ============================================================

struct Struct_Sorting_Pick_Request
{
    float X_mm =
        0.0f;

    float Y_mm =
        0.0f;

    // 机械XY坐标系中的夹爪闭合轴方向，deg，180°周期。
    float Grasp_Axis_Deg =
        0.0f;

    // 视觉估计的“下降前夹爪目标预开口”，单位mm。
    // Valid=false 时 HoldingJaw 保持旧固定预收缩角。
    float Preopen_mm =
        0.0f;

    bool Preopen_Valid =
        false;

    uint32_t Sequence =
        0;
};


// ============================================================
// 第二阶段任务
// ============================================================

struct Struct_Sorting_Calib_Request
{
    float X_mm = 0.0f;
    float Y_mm = 0.0f;
    uint32_t Sequence = 0;
};

struct Struct_Sorting_Yaw_Calib_Request
{
    // 直接的Yaw舵机命令角，deg，不是视觉闭合轴角。
    float Servo_Command_Deg = 0.0f;
    uint32_t Sequence = 0;
};


struct Struct_Sorting_Result_Request
{
    uint8_t Box_ID =
        0;

    uint32_t Sequence =
        0;
};


// ============================================================
// 握手状态机
// ============================================================

class Class_Sorting_Handshake
{
public:

    void Init(
        Class_Sorting_Vision_UART *__Communication
    );


    // ========================================================
    // 周期处理接收队列
    // ========================================================

    void Process();


    // ========================================================
    // STM32实体按键：
    // 开始自动分拣
    // ========================================================

    bool Request_Start_Work();


    // ========================================================
    // STM32实体按键：
    // 停止自动分拣
    // ========================================================

    bool Request_Stop_Work();


    // ========================================================
    // 机械层取顶部识别任务
    // ========================================================

    bool Pop_Pick_Request(
        Struct_Sorting_Pick_Request &Request_Out
    );


    // ========================================================
    // 机械层取侧面识别结果
    // ========================================================

    bool Pop_Sort_Request(
        Struct_Sorting_Result_Request &Request_Out
    );

    // ========================================================
    // 识别台空载恢复
    //
    // Pop_Retry_Pick_Request(): 机械层收到后执行 Recognition -> Disable。
    // Notify_Retry_Pick_Ready(): 只有真正回到 Disable 后调用；该函数
    // 可靠发送 MCU_RETRY_PICK_READY，并把握手重新切回 WAIT_PICK_TARGET。
    // ========================================================

    bool Pop_Retry_Pick_Request();

    bool Notify_Retry_Pick_Ready();

    // ========================================================
    // 储物机构独立测试
    //
    // 仅STOPPED状态接受 STORAGE_TEST(Box_ID)，
    // 不改变正式自动分拣握手状态。
    // ========================================================

    bool Pop_Storage_Test_Request(
        uint8_t &Box_ID_Out
    );

    bool Notify_Storage_Test_Done(
        const uint8_t Box_ID
    );

    bool Pop_Calib_Request(
        Struct_Sorting_Calib_Request &Request_Out
    );

    bool Notify_Calib_Reached(
        const float X_mm,
        const float Y_mm
    );

    bool Pop_Yaw_Calib_Request(
        Struct_Sorting_Yaw_Calib_Request &Request_Out
    );

    bool Notify_Yaw_Calib_Reached(
        const float Servo_Command_Deg
    );


    // ========================================================
    // 第一阶段机械动作完成
    // ========================================================

    bool Notify_Recognition_Ready();


    // ========================================================
    // 第二阶段机械投放完成
    // ========================================================

    bool Notify_Sort_Done();


    // ========================================================
    // 机械层获取STOP请求
    // ========================================================

    bool Pop_Stop_Request(bool *Explicit_Recovery = nullptr);


    // ========================================================
    // Jetson主动STOP的“安全完成ACK”
    //
    // STOP命令刚收到时不立刻回MCU_STOP_ACK。
    // 机械层确认停止/储物复位后才应答；故障停稳可确认STOP，
    // 但START仍单独要求原点有效、真实待机及故障清除。
    // 才调用本函数发送MCU_STOP_ACK。
    // ========================================================

    bool Notify_Stop_Ready();


    inline bool Get_Stop_Ack_Pending() const
    {
        return Stop_Ack_Pending;
    }

    bool Get_Storage_Test_In_Progress() const { return Storage_Test_In_Progress; }


    // ========================================================
    // 错误处理
    // ========================================================

    void Notify_Error(
        const uint8_t &Error_Code =
            0
    );


    // ========================================================
    // 强制回到未启动状态
    // ========================================================

    void Reset_To_Stopped();


    // ========================================================
    // 兼容旧代码
    //
    // 原来的Reset_To_Idle现在等价于STOPPED。
    // ========================================================

    inline void Reset_To_Idle()
    {
        Reset_To_Stopped();
    }


    // ========================================================
    // Getter
    // ========================================================

    inline Enum_Sorting_Handshake_State
    Get_State() const
    {
        return State;
    }


    inline bool
    Get_Running() const
    {
        return
            State !=
                Sorting_Handshake_State_STOPPED
            &&
            State !=
                Sorting_Handshake_State_ERROR
            &&
            State !=
                Sorting_Handshake_State_CALIB_EXECUTING
            &&
            State !=
                Sorting_Handshake_State_CALIB_YAW_EXECUTING;
    }


    inline const Struct_Sorting_Pick_Request &
    Get_Current_Pick() const
    {
        return Current_Pick;
    }


    inline const Struct_Sorting_Result_Request &
    Get_Current_Sort() const
    {
        return Current_Sort;
    }


    inline uint8_t
    Get_Last_Completed_Box() const
    {
        return Last_Completed_Box;
    }


private:

    void Handle_Frame(
        const Struct_Sorting_Vision_UART_Data &Data
    );


    bool Is_Same_Pick(
        const Struct_Sorting_Vision_UART_Data &Data
    ) const;


    Class_Sorting_Vision_UART *Communication =
        nullptr;


    Enum_Sorting_Handshake_State State =
        Sorting_Handshake_State_STOPPED;


    Struct_Sorting_Pick_Request
        Current_Pick{};

    // PICK_PREOPEN 与 PICK_TARGET 分帧发送。
    // 这里只保存“下一件”的预开口；PICK_TARGET锁定后复制并清空，防止串件。
    float Pending_Preopen_mm = 0.0f;
    bool Pending_Preopen_Valid = false;

    Struct_Sorting_Result_Request
        Current_Sort{};

    Struct_Sorting_Calib_Request
        Current_Calib{};

    Struct_Sorting_Yaw_Calib_Request
        Current_Yaw_Calib{};


    bool Pick_Request_Pending =
        false;


    bool Sort_Request_Pending =
        false;

    bool Retry_Pick_Request_Pending =
        false;

    bool Storage_Test_Request_Pending =
        false;

    bool Storage_Test_In_Progress =
        false;

    uint8_t Storage_Test_Box_ID =
        0U;

    bool Calib_Request_Pending =
        false;

    bool Yaw_Calib_Request_Pending =
        false;


    bool Stop_Request_Pending =
        false;
    bool Explicit_Stop_Request_Pending = false;
    bool Error_Notify_Pending = false;
    uint8_t Pending_Error_Code = 0U;

    // Jetson主动STOP已收到，但机械尚未确认安全回位。
    // 该标志为true期间禁止实体START；安全完成后再发送MCU_STOP_ACK。
    bool Stop_Ack_Pending =
        false;


    uint8_t Last_Completed_Box =
        0;


    bool Last_Sort_Done_Valid =
        false;
};


#endif
