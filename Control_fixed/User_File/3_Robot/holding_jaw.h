//
// Created by Lenovo on 2026/8/28.
//

#ifndef DM02_TEST_HOLDING_JAW_H
#define DM02_TEST_HOLDING_JAW_H

#include <cstdint>

#include "2_Device/Motor/Servo_Motor_UART/dvc_servo_motor_uart.h"
#include "3_Robot/xy_motion.h"
#include "2_Device/TOF050F/dvc_tof050f.h"
#include "claw_preopen_profile.h"


// ============================================================
// 抓取总流程状态
// ============================================================

enum Enum_Grap_Move_Status
{
    // 待机
    Enum_Grap_Move_Disable = 0,

    // X/Y/Yaw移动到目标物体
    Enum_Grap_Move_X_Y_Start,

    // 夹爪稍微预收缩
    Enum_Grap_Move_Pre_Clip,

    // TOF闭环控制Lift下降
    Enum_Grap_Move_Descend,

    // 正式夹紧
    Enum_Grap_Move_Clip,

    // 抓住物体后抬升
    Enum_Grap_Move_Lift,

    // X/Y/Yaw移动到识别托盘
    Enum_Grap_Move_X_Y_Tray,

    // 到识别托盘后下降
    Enum_Grap_Move_Tray_Descend,

    // 释放物体
    Enum_Grap_Move_Release,

    // 释放后抬升
    Enum_Grap_Move_Tray_Lift,

    // 放到识别托盘后，移动到专用识别姿态
    Enum_Grap_Move_Recognition_Position,

    // 安全/STOP时整机回待机位置
    Enum_Grap_Move_Return,

    // 一次抓取流程完成，保持在识别姿态
    Enum_Grap_Move_Finish,

    // XY 反馈或驱动故障。该状态是终态，但绝不代表机构已经安全回零。
    Enum_Grap_Move_Fault,
};


// ============================================================
// 夹爪总成
// ============================================================

// 夹爪内部错误原因。串口错误码由 ita_robot.cpp 中的映射函数转换。
enum Enum_Grap_Error : uint8_t
{
    Enum_Grap_Error_None = 0,
    Enum_Grap_Error_XY_Position_Invalid,
    Enum_Grap_Error_TOF_Invalid,
    Enum_Grap_Error_Target_Invalid,
    Enum_Grap_Error_TOF_Timeout,
    Enum_Grap_Error_XY_Drive_Fault,
    Enum_Grap_Error_Actuator_Timeout,
};

class Class_Holding_Jaw
{
public:

    // ========================================================
    // 舵机
    // ========================================================

    Class_Servo_Motor Yaw_Servo_Motor;

    Class_Servo_Motor Lift_Servo_Motor;

    Class_Servo_Motor Claw_Servo_Motor;


    // ========================================================
    // X/Y步进电机
    // ========================================================

    Class_XY_Motion XY_Motion;


    // ========================================================
    // TOF050F
    //
    // 安装在夹爪上
    // ========================================================

    Class_TOF050F TOF050F;


    // ========================================================
    // 初始化
    // ========================================================

    void Init();


    // ========================================================
    // 1ms任务
    //
    // 负责：
    // Lift下降轨迹
    // TOF距离判断
    // 20ms舵机命令分频
    // ========================================================

    void TIM_1ms_PeriodElapsedCallback();


    // ========================================================
    // 100ms抓取状态机
    // ========================================================

    void TIM_100ms_PeriodElapsedCallback();


    // ========================================================
    // 状态机
    // ========================================================

    void Grap_Move();


    // ========================================================
    // 当前状态输出
    // ========================================================

    void Outputs();


    // ========================================================
    // 外部开始一次抓取
    //
    // Target_X_mm：
    //      目标物体X坐标
    //
    // Target_Y_mm：
    //      目标物体Y坐标
    //
    // Target_Yaw_Angle：
    //      抓取Yaw角度
    //      单位rad
    // ========================================================

    HAL_StatusTypeDef Start_Grap(
        float Target_X_mm,

        float Target_Y_mm,

        float Target_Yaw_Angle,

        float Target_Preopen_mm = 0.0f,

        bool Target_Preopen_Valid = false
    );


    // ========================================================
    // 请求从识别姿态/完成状态安全回到Disable
    //
    // 用于STOP、退出测试等场景。
    // 正常完成一次抓取后不会自动调用。
    // ========================================================

    HAL_StatusTypeDef Request_Return_To_Disable();

    // 仅允许在整机完全停稳后人工确认原点。
    HAL_StatusTypeDef Confirm_XY_Origin_mm(float X_mm, float Y_mm);

    // 停止 XY，但不能松开可能夹持的物体，也不能伪称已经安全回零。
    void Abort_XY_Fault();

    // UART 舵机稳定性依据已发送命令经过的时间，不是舵机反馈。
    // XY 就绪始终依据 CAN 实测位置和已确认的停机状态。
    bool Get_Outputs_Settled() const;
    bool Get_Disable_Ready() const;
    // 自动回零前的安全条件，不要求已经建立 XY 原点。
    bool Get_Homing_Ready() const;
    // STOP can be confirmed without a referenced XY origin, including faults.
    bool Get_Stop_Ready() const;
    // 独立 Yaw 标定会使待机姿态失效，直到执行显式 Return 完成回位。
    // HAL_OK accepts the command; callers still wait Duration_ms before ACK.
    HAL_StatusTypeDef Control_Calibration_Yaw(float Radians, uint16_t Duration_ms);
    HAL_StatusTypeDef Stop_Calibration_Yaw();
    // 启动/停止 XY 驱动器顺序碰撞回零，默认由配置锁关闭。
    HAL_StatusTypeDef Start_XY_Collision_Homing();
    void Stop_XY_Collision_Homing();


    // ========================================================
    // 获取抓取状态
    // ========================================================

    inline Enum_Grap_Move_Status
    Get_Grap_Move_Status() const
    {
        return Grap_Move_Status;
    }


    // ========================================================
    // 判断一次流程是否结束
    // ========================================================

    inline bool Get_Grap_Finished() const
    {
        // 正常流程结束后保持在Finish/识别姿态；
        // 故障或STOP安全回位后会进入Disable。
        return
            Grap_Move_Status == Enum_Grap_Move_Finish ||
            Grap_Move_Status == Enum_Grap_Move_Disable ||
            Grap_Move_Status == Enum_Grap_Move_Fault;
    }

    inline bool Get_Grap_Failed() const
    {
        return Grap_Failed;
    }

    // 当前状态的全部输出指令已成功提交；不是舵机物理到位反馈。
    // START 调用处仍需同时检查 Disable、XY 有效和其他机构状态。
    inline bool Get_Grap_Output_Ready() const
    {
        return Grap_Output_Flag;
    }

    inline Enum_Grap_Error Get_Grap_Error() const
    {
        return Grap_Error;
    }


    // ========================================================
    // TOF当前距离
    // ========================================================

    inline uint16_t
    Get_Grap_TOF_Distance_mm() const
    {
        return Grap_TOF_Distance_mm;
    }


    // ========================================================
    // TOF是否已经达到抓取距离
    // ========================================================

    inline bool
    Get_Grap_TOF_Reached() const
    {
        return Grap_TOF_Reached;
    }


    // ========================================================
    // 当前Lift动态下降目标角度
    // ========================================================

    inline float
    Get_Lift_Descend_Target_Angle() const
    {
        return Lift_Descend_Target_Angle;
    }


private:

    // ========================================================
    // X/Y固定运动速度
    //
    // 单位：mm/s
    // ========================================================

    float X_Move_Speed_mm_s =
        45.0f;

    float Y_Move_Speed_mm_s =
        45.0f;


    // ========================================================
    // 外部输入抓取坐标
    // ======================================== ================

    float Grap_Target_X_mm =
        0.0f;

    float Grap_Target_Y_mm =
        0.0f;

    float Grap_Target_Yaw_Angle =
        0.0f;

    float Grap_Target_Preopen_mm =
        0.0f;

    bool Grap_Target_Preopen_Valid =
        false;

    // 本周期预开口角由 CAD 净开口表提供；初始保持已确认的待机张开角。
    float Grap_Target_Pre_Clip_Angle =
        ClawPreopen::kStandbyAngleDeg * PI / 180.0f;

    bool Grap_Failed = false;

    // 故障原因保留到初始化或下一次抓取成功启动，安全回位不清除。
    // 1ms 下降回调写入，机器人调度读取。
    volatile Enum_Grap_Error Grap_Error = Enum_Grap_Error_None;


    // ========================================================
    // 固定识别托盘位置
    // ========================================================

    float Tray_X_mm =
        0.0f;

    float Tray_Y_mm =
        4.0f;

    float Tray_Yaw_Angle =
        90.0f * PI / 180.0f;


    // ========================================================
    // 识别姿态
    //
    // 物体已经释放到识别托盘后，夹爪机构移动到这里，
    // 供侧面相机识别。与Disable完全独立，后续按实机调。
    //
    // 当前默认先沿用识别托盘附近的安全姿态：
    // X/Y与托盘一致、Yaw保持托盘角、Lift抬起、Claw保持释放。
    // ========================================================

    float Recognition_X_mm =
        -72.0f;

    float Recognition_Y_mm =
        0.0f;

    float Recognition_Yaw_Angle =
        110.0f * PI / 180.0f;

    float Recognition_Lift_Angle =
        -65.0f * PI / 180.0f;

    float Recognition_Claw_Angle =
        ClawPreopen::kStandbyAngleDeg * PI / 180.0f;


    // ========================================================
    // 待机位置
    // ========================================================

    // float Disable_X_mm =
    //     0.0f;
    //
    // float Disable_Y_mm =
    //     0.0f;
    float Disable_X_mm =
        0.0f;

    float Disable_Y_mm =
        0.0f;


    // =================================================
    // 舵机固定位置
    // ========================================================


    // --------------------------------------------------------
    // DISABLE / 待机
    // --------------------------------------------------------

    // 驱动角度接口以P1500为0°，PMOD1合法范围是[-135°, +135°]。
    // 旧值240°会被驱动静默夹到+135°，这里改成显式+135°以保持
    // 现有实际动作不变，同时避免“配置值和真实下发值不一致”。
    // 后续Yaw机械标定后请把它替换成真实待机角。
    // float Yaw_Servo_Motor_Angle_Disable =
    //     135.0f * PI / 180.0f;
    //
    //
    // float Lift_Servo_Motor_Angle_Disable =
    //     -60.0f * PI / 180.0f;
    //
    // float Claw_Servo_Motor_Angle_Disable =
    //     0.0f * PI / 180.0f;
    float Yaw_Servo_Motor_Angle_Disable =
        110.0f * PI / 180.0f;


    float Lift_Servo_Motor_Angle_Disable =
        30.0f * PI / 180.0f;

    float Claw_Servo_Motor_Angle_Disable =
        ClawPreopen::kStandbyAngleDeg * PI / 180.0f;

        // 0.0f * PI / 180.0f;



    // --------------------------------------------------------
    // 预收缩
    // --------------------------------------------------------

    float Claw_Servo_Motor_Angle_Pre_Clip =
        -45 * PI / 180.0f;

        // ClawPreopen::kStandbyAngleDeg * PI / 180.0f;

    // --------------------------------------------------------
    // 新夹爪动态预开口：使用 claw_preopen_profile.h 中的线性公式。
    // -35°闭合、-95°待机张开；角度越负越张开。控制接口仍使用 rad。
    // --------------------------------------------------------
    // 旧版问题：直接把 CAD 两个接触参考点的“绝对间距”当作夹爪净开口。
    // 这样 40° 时仍得到约31.02mm，并把上位机小于37.8mm的命令全部
    // 钳到37.8mm，最终几乎所有小/中物体都会映射到35°附近。
    // 实机现象正是：不同物体预收缩都约36°，几乎闭合。
    //
    // 修正：40° 是当前工程的正式 Clip / 近闭合参考，所以把 40° 时
    // CAD 的基准绝对间距扣掉，只计算“相对闭合姿态增加出来的净开口”。
    //
    // CAD绝对几何：
    //   abs_opening(phi) = D + 2*(ry*cos(phi) + rx*sin(phi))
    //   phi = 40° - servo_deg
    //
    // 40°闭合基准：
    //   closed_baseline = D + 2*ry ≈ 31.0241mm
    //
    // 真正用于视觉物体宽度匹配的有效净开口：
    //   gap = abs_opening(phi) - closed_baseline
    //       = 2*[ry*(cos(phi)-1) + rx*sin(phi)]
    //
    // 修正后参考关系：
    //    0° -> 48.89mm
    //    5° -> 43.77mm
    //   10° -> 38.27mm
    //   15° -> 32.45mm
    //   20° -> 26.34mm
    //   25° -> 19.99mm
    //   30° -> 13.45mm
    //   35° ->  6.77mm
    //   40° ->  0.00mm（正式夹紧）
    //
    // 因此小物体会得到较大的预夹角，大物体会保持更开，且不会再因为
    // 37.8mm 下限导致大量物体全部落在35°附近。
    bool Claw_Dynamic_Preopen_Enabled =
        true;

    // --------------------------------------------------------
    // 识别托盘放料下降角
    //
    // 这个角度只用于：
    // “已经抓住物体 -> 移动到识别托盘 -> 放下物体”。
    //
    // 与第一次抓取时的Lift_Descend_Max_Angle完全独立。
    // 后续可单独调这个值，不会改变抓物体时的最大下降量。
    // --------------------------------------------------------

    float Lift_Servo_Motor_Angle_Recognition_Tray_Descend =
        -45.0f * PI / 180.0f;


    // --------------------------------------------------------
    // 正式夹紧
    // --------------------------------------------------------

    float Claw_Servo_Motor_Angle_Clip =
        ClawPreopen::kClosedAngleDeg * PI / 180.0f;


    // --------------------------------------------------------
    // 抬升
    // --------------------------------------------------------

    float Lift_Servo_Motor_Angle_Lift =
        -65.0f * PI / 180.0f;


    // --------------------------------------------------------
    // 释放
    // --------------------------------------------------------

    float Claw_Servo_Motor_Angle_Release =
        ClawPreopen::kStandbyAngleDeg * PI / 180.0f;


    // ========================================================
    // 舵机目标角度
    // ========================================================

    float Yaw_Servo_Motor_Target_Angle =
        0.0f;

    float Lift_Servo_Motor_Target_Angle =
        0.0f;

    float Claw_Servo_Motor_Target_Angle =
        ClawPreopen::kStandbyAngleDeg * PI / 180.0f;


    // ========================================================
    // TOF闭环下降参数
    // ========================================================


    // --------------------------------------------------------
    // 抓取停止距离
    //
    // TOF检测到：
    //
    // Distance <= 30mm
    //
    // 就停止继续下降。
    //
    // 这个后面根据真实夹爪结构调。
    // --------------------------------------------------------

    uint16_t Grap_TOF_Trigger_Distance_mm =
        5;


    // --------------------------------------------------------
    // 连续多少个“新TOF数据”满足阈值
    // 才真正认为到位
    //
    // 2帧约200ms
    // --------------------------------------------------------

    static constexpr uint8_t
    Grap_TOF_Stable_Required =
        2;


    // --------------------------------------------------------
    // Lift下降速度
    //
    // 单位：rad/s
    //
    // 当前10°/s
    // --------------------------------------------------------

    float Lift_Descend_Speed =
        30.0f *
        PI / 180.0f;


    // --------------------------------------------------------
    // Lift下降机械极限
    //
    // 当前-5°
    // --------------------------------------------------------

    float Lift_Descend_Max_Angle =
        15.0f *
        PI / 180.0f;


    // --------------------------------------------------------
    // 动态目标角度
    // --------------------------------------------------------

    float Lift_Descend_Target_Angle =
        -65.0f *
        PI / 180.0f;


    // --------------------------------------------------------
    // 每多少ms向舵机发送一次目标
    //
    // 1ms任务中分频
    // --------------------------------------------------------

    static constexpr uint16_t
    Lift_Control_Send_Period_ms =
        20;


    // --------------------------------------------------------
    // 舵机每次命令的执行时间
    //
    // 单位ms
    //
    // 20ms更新一次目标，
    // 给50ms运动时间，让舵机连续追踪。
    // --------------------------------------------------------

    static constexpr uint16_t
    Lift_Control_Move_Time_ms =
        50;


    // --------------------------------------------------------
    // TOF数据最大允许陈旧时间
    //
    // 如果250ms没有新的TOF数据：
    // 停止继续下降。
    // --------------------------------------------------------

    static constexpr uint16_t
    TOF_Data_Timeout_ms =
        250;


    // ========================================================
    // TOF运行变量
    // ========================================================

    volatile uint16_t Grap_TOF_Distance_mm =
        0;


    volatile bool Grap_TOF_Reached =
        false;


    // Lift已经到下降软件下限，并且最终下限角度命令已成功发送。
    // true时即使TOF尚未达到阈值，也进入Clip执行兜底夹取。
    volatile bool Grap_Lift_Limit_Commanded =
        false;


    volatile uint8_t Grap_TOF_Stable_Count =
        0;


    // 上一次已经处理过的TOF Rx Count
    volatile uint32_t Grap_TOF_Last_Rx_Count =
        0;


    // 距离数据多久没有刷新
    volatile uint16_t Grap_TOF_No_Update_Count_ms =
        0;


    // 20ms发送分频
    volatile uint16_t Lift_Control_Send_Count =
        0;


    // ========================================================
    // 抓取状态
    // ========================================================

    volatile Enum_Grap_Move_Status Grap_Move_Status =
        Enum_Grap_Move_Disable;


    // ========================================================
    // 100ms状态持续计数
    // ========================================================

    uint16_t Grap_Move_Count =
        0;


    // ========================================================
    // 当前状态输出是否已经发送完成
    // ========================================================

    bool Grap_Output_Flag =
        false;


    // ========================================================
    // 多输出状态发送步骤
    // ========================================================

    uint8_t Grap_Output_Step =
        0;

    // 包含每个 XY 阶段的命令提交重试、移动时间和舵机动作时间。
    uint32_t Grap_State_Start_Tick = 0U;
    static constexpr uint32_t XY_State_Timeout_ms = 30000U;
    static constexpr uint32_t Servo_Move_Time_ms = 500U;
    static constexpr uint32_t Servo_Stop_Settle_ms = 20U;
    uint32_t Grap_Output_Complete_Tick = 0U;
    uint32_t Return_Lift_Command_Tick = 0U;
    uint32_t Grap_Lift_Limit_Command_Tick = 0U;
    uint32_t Fault_Stop_Complete_Tick = 0U;
    bool Fault_Lift_Stop_Pending = false;
    bool Fault_Yaw_Stop_Pending = false;
    bool Calibration_Yaw_Override = false;
    uint32_t Calibration_Yaw_Command_Tick = 0U;
    uint32_t Calibration_Yaw_Settle_ms = 0U;


    // ========================================================
    // 内部函数
    // ========================================================

    void Set_Grap_Move_Status(
        Enum_Grap_Move_Status New_Status
    );


    bool XY_Move_Finished() const;
    bool XY_At_mm(float X_mm, float Y_mm) const;
    bool Motion_Configuration_Valid() const;
    void Complete_Outputs();
    void Abort_Grap_Fault(Enum_Grap_Error Reason);
    void Service_Fault_Servo_Stop();
};


#endif // DM02_TEST_HOLDING_JAW_H
