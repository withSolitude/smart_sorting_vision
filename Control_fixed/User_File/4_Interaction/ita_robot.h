//
// Created by Lenovo on 2026/1/12.
//

#ifndef TEST_ROBOWAKER_ITA_ROBOT_H
#define TEST_ROBOWAKER_ITA_ROBOT_H


#include "1_Middleware/2_Algorithm/Slope/alg_slope.h"
#include "1_Middleware/2_Algorithm/Timer/alg_timer.h"
#include "2_Device/Motor/Servo_Motor_UART/dvc_servo_motor_uart.h"
#include "2_Device/PC_Communication/dvc_sorting_vision_uart_v2.h"
#include "3_Robot/holding_jaw.h"
#include "4_Interaction/ita_sorting_handshake.h"
#include "2_Device/PC_Communication/dvc_sorting_vision_uart_v2.h"
#include "3_Robot/sorting_storage.h"

class Class_Robot
{
public:

    // ========================================================
    // 现有夹爪机构
    // ========================================================

    Class_Holding_Jaw
        Holding_Jaw;

    // ========================================================
    // 储物4310和舵机
    // ========================================================
    Class_Sorting_Storage
    Sorting_Storage;


    // ========================================================
    // Jetson UART通信
    // ========================================================

    Class_Sorting_Vision_UART
        Sorting_Vision_UART;


    // ========================================================
    // 分拣通信状态机
    // ========================================================

    Class_Sorting_Handshake
        Sorting_Handshake;


    // ========================================================
    // 原初始化
    // ========================================================

    void Init();


    // ========================================================
    // 初始化Jetson专用UART
    //
    // 例如：
    //
    // Robot.Init_Sorting_Communication(&huart1);
    //
    // 具体huart由你的硬件决定。
    // ========================================================

    void Init_Sorting_Communication(
        UART_HandleTypeDef *UART_Handler
    );


    // ========================================================
    // 实体START请求（短按）
    //
    // 与旧Toggle语义不同：
    // - STOPPED时只“锁存”一次START请求；
    // - 如果机构尚未Ready，请求不会丢失；
    // - 后续10ms调度中一旦全部Ready会自动真正START；
    // - RUNNING时短按由按键层直接忽略。
    // ========================================================

    bool Request_Sorting_Start();


    // 兼容旧测试接口。正式实体按键不再调用Toggle。
    bool Toggle_Sorting_Work();


    bool Start_Sorting_Work();


    bool Stop_Sorting_Work();

    // 上电后由操作员把滑台放到已知 XY 坐标。
    // 仅允许在停机且整机所有机构空闲时确认。
    HAL_StatusTypeDef Confirm_XY_Origin_mm(float X_mm = 0.0f, float Y_mm = 0.0f);
    HAL_StatusTypeDef Start_XY_Collision_Homing();
    void Stop_XY_Collision_Homing();


    inline bool Get_Sorting_Start_Request_Pending() const
    {
        return Sorting_Start_Request_Pending;
    }

    bool Get_Calibration_Active() const { return Calib_Action_Active || Yaw_Calib_Action_Active; }

    bool Get_Mechanism_Action_Active() const
    {
        return Get_Sorting_Running() || Sorting_Start_Request_Pending ||
            Sorting_Handshake.Get_State() == Sorting_Handshake_State_ERROR ||
            Sorting_Storage.Get_Fault() || Holding_Jaw.XY_Motion.Get_Fault() ||
            Holding_Jaw.Get_Grap_Move_Status() == Enum_Grap_Move_Fault ||
            Get_Calibration_Active() || Storage_Test_Start_Pending ||
            Storage_Test_Action_Active || Storage_Test_Done_Notify_Pending ||
            Yaw_Stop_Pending || Retry_Pick_Return_Active || Stop_Waiting_Safe_Disable ||
            Holding_Jaw.XY_Motion.Get_Homing_Active() ||
            (Holding_Jaw.Get_Grap_Move_Status() != Enum_Grap_Move_Disable &&
             Holding_Jaw.Get_Grap_Move_Status() != Enum_Grap_Move_Fault);
    }


    // ========================================================
    // 旧通信仿真兼容接口。正式完成由Storage完成事件和真实待机互锁处理。
    // ========================================================

    bool Notify_Sort_Mechanism_Done();


    // ========================================================
    // 当前目标储物盒
    // ========================================================

    inline uint8_t
    Get_Sort_Target_Box_ID() const
    {
        return
            Sort_Target_Box_ID;
    }


    inline bool
    Get_Sort_Action_Active() const
    {
        return
            Sort_Action_Active;
    }


    inline bool
    Get_Sorting_Running() const
    {
        return
            Sorting_Handshake
                .Get_Running();
    }


    // ========================================================
    // 原函数
    // ========================================================

    void Loop();


    void TIM_1000ms_Alive_PeriodElapsedCallback();


    void TIM_200ms_Alive_PeriodElapsedCallback();


    void TIM_100ms_Alive_PeriodElapsedCallback();


    void TIM_100ms_Calculate_Callback();


    void TIM_10ms_Calculate_PeriodElapsedCallback();


    void TIM_2ms_Calculate_PeriodElapsedCallback();


    void TIM_1ms_Calculate_Callback();


private:

    // STOP必须先于任何新机械任务消费。
    void Handle_Stop_Request();

    // 取消等待并上报反馈/运动故障，不能伪称机构已经回到原点。
    bool Service_XY_Fault();
    bool XY_Fault_Reported = false;
    bool Storage_Fault_Reported = false;
    bool Storage_Recovery_Requested = false;
    // 视觉链路连续掉线时只触发一次安全停止，重连后允许重新握手继续比赛。
    uint8_t Vision_Link_Lost_Seconds = 0U;
    bool Vision_Link_Lost_Stop_Requested = false;
    void Service_Storage_Recovery();

    // 统一判断当前机械是否已经回到“允许START”的安全态。
    bool Is_Sorting_Start_Ready();

    // STOP synchronization needs idle mechanisms; START additionally needs
    // 启动还必须满足机械尺度已标定，且已经人工建立 XY 原点。
    bool Is_Sorting_Stop_Ready();

    // Jetson主动STOP收到后，不在通信层立即ACK。
    // 只有Is_Sorting_Stop_Ready()成立时才发送MCU_STOP_ACK，
    // 作为上下位机上电同步完成边界。
    void Service_Sorting_Stop_Ready_Ack();

    // 实体短按START采用“锁存请求 + Ready后自动启动”。
    // 该函数在10ms调度中反复尝试，直到Start_Sorting_Work()成功。
    void Service_Sorting_Start_Request();

    // Task_Loop前台写入、10ms定时调度读取，因此使用volatile。
    volatile bool Sorting_Start_Request_Pending = false;

    // ========================================================
    // 顶部识别结果已经收到，
    // 但是Holding_Jaw还暂时不能Start_Grap时，
    // 将任务保存在这里不断重试。
    // ========================================================

    Struct_Sorting_Pick_Request
        Pending_Pick_Request{};


    bool Pick_Start_Pending =
        false;


    bool Pick_Action_Active =
        false;

    uint16_t Pick_Start_Wait_Count = 0;


    // ========================================================
    // 动态抓取Yaw机械标定
    //
    // Jetson发送的是“夹爪闭合轴在机械XY坐标系中的方向”，不是舵机原始角。
    // Yaw机构存在齿轮/传动时，舵机角和夹爪实际Yaw不一定是1:1，
    // 因此不能简单 servo_yaw = vision_yaw。
    //
    // 建议实机选一个参考姿态：
    //   Reference_Closing_Axis_Robot_Deg : 该姿态下夹爪闭合轴在机械XY中的方向
    //   Reference_Servo_Command_Deg      : 该姿态对应的舵机命令角
    //   Servo_Deg_Per_Gripper_Deg        : 夹爪实际转1°需要舵机转多少°，含正负号
    //
    // 例如直驱同向约为 +1；外啮合齿轮通常为负号，数值还要乘传动比。
    // 具体值必须实测，不从照片/视觉代码猜。
    // ========================================================
    // 2026-09-12 实机Yaw标定（16点，舵机命令覆盖 -60°~+135°）
    // 拟合结果：q = q0 + k * (theta - theta0)
    // theta 为机械XY坐标系中的夹爪闭合轴方向（180°周期）。
    // 标定报告：axis RMSE = 3.487°，max = 6.264°。
    float Pick_Yaw_Reference_Closing_Axis_Robot_Deg = 43.590200f;
    float Pick_Yaw_Reference_Servo_Command_Deg = -0.622672f;
    float Pick_Yaw_Servo_Deg_Per_Gripper_Deg = -1.10205336f;

    // ========================================================
    // branch=1 专用实际夹爪方向修正
    //
    // 单位：夹爪物理闭合轴角度 deg。
    // 只作用于另一物理侧；branch=0完全不受影响。
    //
    // 必须与 Jetson runtime_config.yaml 中：
    // branch1_axis_correction_deg
    // 保持完全相同。
    //
    // 初始为0.0f，实机先按 ±2° 测，再用0.5°步长精调。
    // ========================================================
    float Pick_Yaw_Branch1_Axis_Correction_Deg = 0.0f;

    // 只允许进入本轮实机实际标定覆盖过的舵机区间。
    // 该区间宽195°，大于闭合轴完整180°周期经传动后的约190°，
    // 因而仍能为任意闭合轴方向选择一个等效合法解。
    float Pick_Yaw_Servo_Command_Min_Deg = -135.0f;
    float Pick_Yaw_Servo_Command_Max_Deg =  135.0f;

    // 双边抓取：Yaw_Branch由Jetson显式选择物理180°分支。
    // 0 = 轴向角本身；1 = 与其相差180°的另一物理姿态。
    float Convert_Grasp_Axis_To_Servo_Yaw_Rad(
        float Grasp_Axis_Robot_Deg,
        uint8_t Yaw_Branch
    ) const;


    // ========================================================
    // STOP期间：
    //
    // 如果夹爪正在搬运当前物体，
    // 不突然松爪。
    //
    // 允许当前第一阶段安全完成并回Disable，
    // 但完成后不再给Jetson发送Recognition Ready。
    // ========================================================

    bool Stop_Waiting_Safe_Disable =
        false;


    // ========================================================
    // 识别托盘空载恢复
    //
    // Jetson 发送 RETRY_PICK 后，机械只执行 Recognition -> Disable，
    // 不触发 STOP、不操作储物机构。真正到 Disable 后再通知握手层发送
    // MCU_RETRY_PICK_READY，随后继续下一次顶部抓取。
    // ========================================================

    bool Retry_Pick_Return_Active =
        false;


    // ========================================================
    // 第二阶段
    //
    // 1~6箱号对应DM4310+翻转托盘，完成后还要等待夹爪真实待机。
    // ========================================================

    uint8_t Sort_Target_Box_ID =
        0;


    bool Sort_Action_Active =
        false;

    // 机械投放完成后，若DONE暂时不能入TX队列，保留该事件重试，
    // 绝不重新执行机械动作。
    bool Sort_Done_Notify_Pending = false;

    Struct_Sorting_Result_Request Pending_Sort_Request{};
    bool Sort_Start_Pending = false;
    uint16_t Sort_Start_Wait_Count = 0;

    // XY标定独立运动状态
    bool Calib_Action_Active = false;
    bool Calib_Start_Pending = false;
    float Calib_Target_X_mm = 0.0f;
    float Calib_Target_Y_mm = 0.0f;
    uint16_t Calib_Move_Count = 0;
    uint8_t Calib_Stable_Count = 0;
    static constexpr uint16_t Calib_Timeout_10ms = 1500;
    static constexpr uint8_t Calib_Stable_Required = 15;
    static constexpr float Calib_Position_Tolerance_mm = 0.30f;
    static constexpr float Calib_Move_Speed_mm_s = 35.0f;

    // Yaw标定独立运动状态。当前众灵舵机没有在主循环中持续回读实际角度，
    // 因此REACHED按“命令成功发送 + 设定运动时间 + 稳定余量”判定。
    bool Yaw_Calib_Action_Active = false;
    bool Yaw_Calib_Start_Pending = false;
    bool Yaw_Stop_Pending = false;
    float Yaw_Calib_Target_Deg = 0.0f;
    uint16_t Yaw_Calib_Count_10ms = 0;
    static constexpr uint16_t Yaw_Calib_Move_Time_ms = 700;
    static constexpr uint16_t Yaw_Calib_Reach_Count_10ms = 100; // 约1.0s
    static constexpr uint16_t Yaw_Calib_Timeout_10ms = 200;     // 2.0s


    // ========================================================
    // 储物机构独立测试
    //
    // STORAGE_TEST 只允许正式自动分拣 STOPPED 时使用。
    // 用于“侧面识别 -> Box_ID -> 4310/翻转托盘”的后半链路联调。
    // ========================================================

    bool Storage_Test_Start_Pending = false;
    bool Storage_Test_Action_Active = false;
    bool Storage_Test_Done_Notify_Pending = false;
    uint8_t Storage_Test_Box_ID = 0U;
    uint16_t Storage_Test_Start_Wait_Count = 0U;


    // ========================================================
    // 纯通信联调开关
    //
    // false:
    //      正式模式。
    //      收到Box_ID后执行真实4310/翻转托盘流程。
    //
    // true:
    //      临时联调。
    //      等1秒自动模拟SORT_DONE。
    //
    // 正式机械接入后保持false。
    // ========================================================

    static constexpr bool
        Sort_Action_Simulation_Enable =
            false;


    uint16_t Sort_Action_Simulation_Count =
        0;


    static constexpr uint16_t
        Sort_Action_Simulation_Time_10ms =
            100;
};


#endif
