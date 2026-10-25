//
// Created for Smart Sorting System
// Rewritten following holding_jaw style
//

#ifndef DM02_TEST_SORTING_STORAGE_H
#define DM02_TEST_SORTING_STORAGE_H


#include <cstdint>
#include <cmath>

#include "2_Device/Motor/Motor_DM/dvc_motor_dm.h"
#include "2_Device/Motor/Servo_Motor_UART/dvc_servo_motor_uart.h"
#include "arm_math.h"


// ============================================================
// 储物盒投放机构状态
// ============================================================

enum Enum_Sorting_Storage_Status : uint8_t
{
    // 待机
    Sorting_Storage_Status_DISABLE = 0,

    // 上电后等待DM4310真正在线并成功使能
    Sorting_Storage_Status_WAIT_DM_READY,

    // 上电同步完成后，DM4310回到待机方向
    Sorting_Storage_Status_DM_STARTUP_RETURN,

    // DM4310转到目标储物盒
    Sorting_Storage_Status_DM_MOVE_TO_BOX,

    // 托盘翻转
    Sorting_Storage_Status_TRAY_FLIP,

    // 保持翻转，等待物体落下
    Sorting_Storage_Status_TRAY_HOLD,

    // 托盘恢复正常位置
    Sorting_Storage_Status_TRAY_RETURN,

    // DM4310回待机位置
    Sorting_Storage_Status_DM_RETURN,

    // STOP：先恢复托盘
    Sorting_Storage_Status_STOP_TRAY_RETURN,

    // STOP：再让DM4310回待机位置
    Sorting_Storage_Status_STOP_DM_RETURN,

    // 错误
    Sorting_Storage_Status_ERROR,
};


// ============================================================
// 储物盒 + 翻转托盘
// ============================================================

class Class_Sorting_Storage
{
public:

    // ========================================================
    // 达妙4310
    // ========================================================

    Class_Motor_DM_Normal DM4310;


    // ========================================================
    // 托盘翻转舵机
    // ========================================================

    Class_Servo_Motor Tray_Flip_Servo_Motor;


    // ========================================================
    // 初始化
    // ========================================================

    void Init(
        const FDCAN_HandleTypeDef *DM_CAN_Handler,

        const uint8_t DM_CAN_Rx_ID,

        const uint8_t DM_CAN_Tx_ID,

        UART_HandleTypeDef *Servo_UART_Handler,

        const uint8_t Servo_ID
    );


    // ========================================================
    // 10ms任务
    //
    // 和Holding_Jaw一致：
    //
    // 先执行当前状态输出
    // 再判断当前状态是否完成
    // ========================================================

    void TIM_10ms_PeriodElapsedCallback();


    // ========================================================
    // 状态机
    // ========================================================

    void Storage_Move();


    // ========================================================
    // 当前状态输出
    // ========================================================

    void Outputs();


    // ========================================================
    // 2ms运行DM4310双环PID并用MIT模式下发力矩
    // ========================================================

    void TIM_2ms_PeriodElapsedCallback();


    // ========================================================
    // 100ms在线检测
    // ========================================================

    void TIM_100ms_Alive_PeriodElapsedCallback();


    // ========================================================
    // 开始一次投放
    //
    // Box_ID = 1 ~ 6
    // ========================================================

    HAL_StatusTypeDef Start_Sort(
        const uint8_t Box_ID
    );


    // ========================================================
    // STOP / 强制安全回Disable
    // ========================================================

    void Request_Disable();

    // 安全待机必须同时满足 DM 反馈新鲜和托盘回位指令的动作时间。
    // UART 舵机没有实测位置应答，因此这里只能依据已发送命令和时间判断。
    bool Get_Disable_Ready();
    inline bool Get_Fault() const { return Fault_Latched; }
    // Explicit operator recovery; communication returning alone never clears it.
    bool Clear_Fault();


    // ========================================================
    // CAN回调
    // ========================================================

    void CAN_RxCpltCallback(
        FDCAN_HandleTypeDef *hfdcan,

        const FDCAN_RxHeaderTypeDef &Header
    );


    // ========================================================
    // UART回调
    // ========================================================

    void UART_RxCpltCallback(
        UART_HandleTypeDef *huart,

        uint8_t *Buffer,

        uint16_t Length
    );


    // ========================================================
    // 一次投放完成Pending
    //
    // 读取一次以后自动清零
    // ========================================================

    bool Pop_Sort_Finished();


    // ========================================================
    // 当前是否正在动作
    // ========================================================

    inline bool Get_Busy() const
    {
        return
            Storage_Status !=
            Sorting_Storage_Status_DISABLE;
    }


    // ========================================================
    // 当前状态
    // ========================================================

    inline Enum_Sorting_Storage_Status
    Get_Status() const
    {
        return Storage_Status;
    }


    // ========================================================
    // 当前目标Box
    // ========================================================

    inline uint8_t
    Get_Target_Box_ID() const
    {
        return Target_Box_ID;
    }


    // ========================================================
    // DM4310当前实际机械角度
    //
    // 这里不再返回普通反馈POS(p_m)，而是返回寄存器0x51的
    // 输出轴编码器xout经过方向/零偏修正后的0~2π机械角。
    // ========================================================

    inline float
    Get_DM4310_Now_Angle()
    {
        return Get_DM4310_Output_Angle();
    }

    // 普通反馈帧里的POS，保留仅用于对比调试。
    inline float
    Get_DM4310_Raw_Feedback_Angle()
    {
        return DM4310.Get_Now_Angle();
    }

    // 原始xout寄存器值。
    inline float
    Get_DM4310_Raw_Output_Shaft_Angle()
    {
        return DM4310.Get_Output_Shaft_Angle();
    }

    inline bool
    Get_DM4310_Output_Shaft_Valid()
    {
        return DM4310.Get_Output_Shaft_Angle_Valid();
    }


    // ========================================================
    // DM4310当前目标角度
    // ========================================================

    inline float
    Get_DM4310_Target_Angle() const
    {
        return DM4310_Target_Angle;
    }


    // ========================================================
    // DM4310双环PID调试量
    // ========================================================

    inline float Get_DM4310_PID_Target_Omega() const
    {
        return DM4310_PID_Target_Omega;
    }

    inline float Get_DM4310_PID_Torque_Command() const
    {
        return DM4310_PID_Torque_Command;
    }


    // ========================================================
    // DM4310是否已经完成上电同步
    // ========================================================

    inline bool
    Get_DM4310_Startup_Ready() const
    {
        return DM4310_Startup_Ready;
    }


    // ========================================================
    // 修改单个储物盒标定角度
    //
    // Box_ID = 1 ~ 6
    // Angle_Rad单位rad
    // ========================================================

    void Set_Box_Angle(
        const uint8_t Box_ID,

        const float Angle_Rad
    );


    // ========================================================
    // 修改DM待机方向
    //
    // 注意：
    // 这里只修改目标参数，不执行“保存零点”。
    // DM机械零点需要单独标定并在电机内保存。
    // ========================================================

    inline void Set_DM4310_Disable_Angle(
        const float Angle_Rad
    )
    {
        if (!std::isfinite(Angle_Rad)) return;
        DM4310_Disable_Angle =
            Normalize_Base_Angle(Angle_Rad);
    }

    // 如果机械零位处读取到的xout并不是0，可把该xout值填到这里。
    // 默认0表示已经通过达妙“保存位置零点”把机械待机位保存为0。
    inline void Set_DM4310_Output_Zero_Offset(
        const float Offset_Rad
    )
    {
        if (std::isfinite(Offset_Rad)) DM4310_Output_Zero_Offset = Offset_Rad;
    }

    // 输出轴编码器正方向与转盘定义相反时设为-1，默认+1。
    inline void Set_DM4310_Output_Direction(
        const float Direction
    )
    {
        if (std::isfinite(Direction) && Direction != 0.0f)
            DM4310_Output_Direction = Direction > 0.0f ? 1.0f : -1.0f;
    }


    // ========================================================
    // 修改托盘正常 / 翻转 PWM
    //
    // 这里只修改标定值，不立即动作。
    // 实际调试请使用下面的Debug接口。
    // ========================================================

    inline void Set_Tray_Disable_PWM(
        const uint16_t PWM
    )
    {
        if (PWM < 500U || PWM > 2500U) return;
        Tray_Servo_PWM_Disable =
            PWM;
    }


    inline void Set_Tray_Flip_PWM(
        const uint16_t PWM
    )
    {
        if (PWM < 500U || PWM > 2500U) return;
        Tray_Servo_PWM_Flip =
            PWM;
    }


    inline uint16_t
    Get_Tray_Disable_PWM() const
    {
        return Tray_Servo_PWM_Disable;
    }


    inline uint16_t
    Get_Tray_Flip_PWM() const
    {
        return Tray_Servo_PWM_Flip;
    }


    // ========================================================
    // 托盘舵机PWM调试接口
    //
    // 允许状态：
    //      DISABLE
    //      WAIT_DM_READY
    //
    // PWM：
    //      舵机底层会自动限制到500 ~ 2500
    //
    // Move_Time_ms：
    //      舵机运动时间
    //
    // 该接口强制发送，即使PWM与上一条命令相同也会发送。
    // ========================================================

    HAL_StatusTypeDef Debug_Set_Tray_PWM(
        const uint16_t PWM,

        const uint16_t Move_Time_ms = 600
    );


    // ========================================================
    // 调试：移动到当前记录的正常位置
    // ========================================================

    HAL_StatusTypeDef Debug_Tray_To_Disable(
        const uint16_t Move_Time_ms = 600
    );


    // ========================================================
    // 调试：移动到当前记录的翻转位置
    // ========================================================

    HAL_StatusTypeDef Debug_Tray_To_Flip(
        const uint16_t Move_Time_ms = 600
    );


    // ========================================================
    // 最近一次成功发送给托盘舵机的PWM
    // ========================================================

    inline uint16_t
    Get_Tray_Last_Command_PWM() const
    {
        return Tray_Last_Command_PWM;
    }


private:


    // ============================================================
    // DM4310上电启动
    // ============================================================

    // 100ms计数
    uint16_t DM4310_Startup_Count_100ms =
        0;

    // 启动重试阶段
    uint8_t DM4310_Startup_Phase =
        0;

    // 调试
    volatile uint32_t DM4310_Startup_Clear_Count =
        0;

    volatile uint32_t DM4310_Startup_Enter_Count =
        0;






    static constexpr float TWO_PI =
        2.0f * PI;


    // ========================================================
    // 硬件句柄
    // ========================================================

    const FDCAN_HandleTypeDef *DM_CAN_Handler =
        nullptr;


    UART_HandleTypeDef *Servo_UART_Handler =
        nullptr;


    uint8_t DM4310_CAN_Rx_ID =
        0;


    // ========================================================
    // 6个储物盒标定角
    //
    // 当前先按照60°均匀分布。
    // 后面实际装机以后逐个标定。
    // ========================================================

    float Box_Angle_Rad[6] =
    {
        0.0f * PI / 180.0f,

        60.0f * PI / 180.0f,

        120.0f * PI / 180.0f,

        180.0f * PI / 180.0f,

        240.0f * PI / 180.0f,

        300.0f * PI / 180.0f,
    };


    // ========================================================
    // DM4310待机方向
    // ========================================================

    float DM4310_Disable_Angle =
        0.0f;


    // ========================================================
    // DM4310 MIT力矩控制 + 自写双环PID
    //
    // 位置外环：角度误差 -> 目标角速度
    // 速度内环：角速度误差 -> 目标力矩
    //
    // MIT电机内部Kp/Kd固定为0，只使用Torque前馈通道，
    // 因此真正的位置/速度闭环都在STM32里完成。
    // ========================================================

    static constexpr float DM4310_PID_DT =
        0.002f;

    // 外环最大目标速度，先保守设为1.0rad/s。
    float DM4310_Move_Omega =
        1.5f;

    // 位置外环PID。初值以P为主，D直接使用-now_omega抑制过冲。
    float DM4310_Pos_Kp =
        2.5f;

    float DM4310_Pos_Ki =
        0.0f;

    float DM4310_Pos_Kd =
        0.08f;

    // 速度内环PID。
    float DM4310_Omega_Kp =
        1.20f;

    float DM4310_Omega_Ki =
        3.20f;

    float DM4310_Omega_Kd =
        0.0f;

    // 先把最大力矩限制在2Nm，实机确认后再逐步提高。
    float DM4310_PID_Torque_Limit =
        4.0f;

    // 每2ms最大力矩变化量，避免上电/切目标时扭矩阶跃。
    float DM4310_PID_Torque_Slew_Per_2ms =
        0.15f;

    // 积分限幅，防止长时间堵转/到不了目标时积分饱和。
    float DM4310_Pos_Integral_Limit =
        0.50f;

    float DM4310_Omega_Integral_Limit =
        1.50f;

    // PID运行状态。
    float DM4310_Pos_Integral =
        0.0f;

    float DM4310_Omega_Integral =
        0.0f;

    float DM4310_Pre_Omega_Error =
        0.0f;

    bool DM4310_Pre_Omega_Error_Valid =
        false;

    float DM4310_PID_Target_Omega =
        0.0f;

    float DM4310_PID_Torque_Command =
        0.0f;

    // ========================================================
    // 输出轴编码器xout作为真正位置闭环
    // ========================================================

    // 机械零位对应的xout原始值。若电机已保存零点，保持0即可。
    float DM4310_Output_Zero_Offset =
        0.0f;

    // xout正方向到转盘角正方向的符号映射。
    float DM4310_Output_Direction =
        1.0f;

    // xout寄存器更新计数/失联保护。
    uint32_t DM4310_Xout_Last_Update_Count =
        0U;

    uint16_t DM4310_Xout_Stale_2ms_Count =
        0U;

    // 100ms没有新的xout就认为位置反馈失效，强制0力矩。
    static constexpr uint16_t
    DM4310_Xout_Stale_Limit_2ms =
        50U;


    // ========================================================
    // DM4310到位条件
    //
    // 角度误差 <= 2°
    // 速度 <= 0.15 rad/s
    // ========================================================

    float DM4310_Angle_Tolerance =
        2.0f *
        PI /
        180.0f;


    float DM4310_Omega_Tolerance =
        0.15f;


    // ========================================================
    // 连续5个10ms满足才算真正到位
    // ========================================================

    static constexpr uint8_t
    DM4310_Stable_Required =
        5;


    uint8_t DM4310_Stable_Count =
        0;


    // ========================================================
    // DM运动超时
    //
    // 500 × 10ms = 5s
    // ========================================================

    static constexpr uint16_t
    DM4310_Move_Timeout_Count =
        500;


    // ========================================================
    // DM上电同步标志
    //
    // false时：
    // 不允许2ms力矩PID开始工作。
    //
    // 等收到真实反馈并确认电机Enable后，
    // 再使用“圆周角误差”判断是否需要回待机方向。
    // ========================================================

    bool DM4310_Startup_Ready =
        false;


    // ========================================================
    // 托盘舵机标定PWM
    // ========================================================

    uint16_t Tray_Servo_PWM_Disable =
        1480;


    uint16_t Tray_Servo_PWM_Flip =
        900;


    // ========================================================
    // 托盘动作时间
    // ========================================================

    static constexpr uint16_t
    Tray_Flip_Move_Time_ms =
        200;


    static constexpr uint16_t
    Tray_Return_Move_Time_ms =
        200;


    static constexpr uint16_t
    Tray_Move_Safe_Time_ms =
        200;


    static constexpr uint16_t
    Tray_Hold_Time_ms =
        200 ;


    // ========================================================
    // 托盘最后一次成功发送的PWM
    //
    // 用于阻止：
    // TRAY_RETURN已经发过一次回位，
    // 随后进入DISABLE又重复发一次相同回位命令。
    // ========================================================

    uint16_t Tray_Last_Command_PWM =
        0;


    bool Tray_Last_Command_Valid =
        false;

    uint32_t Tray_Last_Command_Tick = 0U;
    uint16_t Tray_Last_Command_Move_Time_ms = 0U;
    uint32_t State_Start_Tick = 0U;
    static constexpr uint32_t Output_Submit_Timeout_ms = 3000U;
    static constexpr uint32_t DM_Feedback_Timeout_ms = 100U;
    bool Fault_Latched = false;
    bool Fault_Recovery_Requested = false;
    bool Xout_Clock_Valid = false;
    bool Feedback_Clock_Valid = false;
    uint32_t Xout_Seen_Count = 0U;
    uint32_t Feedback_Seen_Count = 0U;
    uint32_t Xout_Last_Tick = 0U;
    uint32_t Feedback_Last_Tick = 0U;
    uint32_t Stable_Xout_Last_Count = 0U;


    // ========================================================
    // 状态运行变量
    // ========================================================

    volatile Enum_Sorting_Storage_Status Storage_Status =
        Sorting_Storage_Status_WAIT_DM_READY;


    uint16_t Storage_Move_Count =
        0;


    bool Storage_Output_Flag =
        false;


    uint8_t Target_Box_ID =
        0;


    float DM4310_Target_Angle =
        0.0f;


    volatile bool Sort_Finished_Pending =
        false;


    // ========================================================
    // 内部函数
    // ========================================================

    void Set_Storage_Status(
        Enum_Sorting_Storage_Status New_Status
    );


    bool DM4310_Target_Reached();
    void Observe_DM4310_Feedback();
    bool DM4310_Feedback_Fresh();
    bool Tray_Disable_Ready() const;


    void Update_DM4310_Stable_Count();


    void Set_DM4310_Target(
        const float Target_Angle
    );


    float Normalize_Base_Angle(
        const float Angle_Rad
    ) const;

    float Normalize_Angle_Error(
        const float Angle_Error_Rad
    ) const;

    float Get_Circular_Angle_Error(
        const float Target_Angle_Rad,
        const float Now_Angle_Rad
    ) const;

    // 使用0x51 xout得到真正机械角度，返回0~2π。
    float Get_DM4310_Output_Angle() const;

    void Reset_DM4310_PID(
        const bool Reset_Torque_Command = true
    );

    void Update_DM4310_PID();

    static float Clamp_Float(
        const float Value,
        const float Min_Value,
        const float Max_Value
    );


    bool Try_Send_Tray_PWM(
        const uint16_t PWM,

        const uint16_t Move_Time_ms,

        const bool Force_Send = false
    );


    void Enter_Error();
};


#endif // DM02_TEST_SORTING_STORAGE_H
