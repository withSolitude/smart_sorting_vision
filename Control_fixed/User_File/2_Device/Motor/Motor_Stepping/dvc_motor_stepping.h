/**
 * @file dvc_motor_stepping.h
 * @brief DM430 步进电机直线轴驱动
 *
 * 控制模式：
 *      PUL + DIR
 *
 * 主要功能：
 *      1. 原始脉冲运动
 *      2. mm / pulse 映射
 *      3. 多圈累计位置
 *      4. 绝对位置运动
 *      5. 相对位置运动
 *      6. 连续速度运动
 *      7. 运行中动态修改速度
 *
 * 加减速规划由外部 Class_Slope 完成。
 */

#ifndef DVC_MOTOR_STEPPING_H
#define DVC_MOTOR_STEPPING_H

#include <cstdint>

#include "tim.h"
#include "stm32h7xx_hal.h"


// ============================================================
// DM430 默认参数
// ============================================================

// DM430有效脉宽
// 手册要求 >= 2us
// 这里使用5us留裕量
#define MOTOR_STEPPING_DM430_PULSE_ACTIVE_TIME_US              5U

// PUL低电平/非有效状态最小时间
#define MOTOR_STEPPING_DM430_PULSE_INACTIVE_MIN_TIME_US        5U

// DIR改变后等待时间
#define MOTOR_STEPPING_DM430_DIRECTION_SETUP_TIME_US           5U

// TIM13/TIM14:
// 240MHz / 240 = 1MHz
#define MOTOR_STEPPING_DM430_DEFAULT_TIMER_TICK_HZ             1000000.0f

// 当前软件GPIO方案限制
#define MOTOR_STEPPING_DM430_DEFAULT_MAX_PULSE_FREQUENCY_HZ    20000.0f


// ============================================================
// 电机状态
// ============================================================

enum Enum_Motor_Stepping_Status : uint8_t
{
    Motor_Stepping_Status_STOP = 0,

    Motor_Stepping_Status_RUNNING,

    Motor_Stepping_Status_PAUSED,
};


// ============================================================
// 运动模式
// ============================================================

enum Enum_Motor_Stepping_Control_Method : uint8_t
{
    Motor_Stepping_Control_Method_STOP = 0,

    // 原始脉冲运动
    Motor_Stepping_Control_Method_PULSES,

    // 相对直线位移
    Motor_Stepping_Control_Method_RELATIVE_MM,

    // 绝对直线位置
    Motor_Stepping_Control_Method_ABSOLUTE_MM,

    // 连续速度
    Motor_Stepping_Control_Method_SPEED_MM_S,
};


// ============================================================
// 方向
// ============================================================

enum Enum_Motor_Stepping_Direction : uint8_t
{
    // XY坐标正方向
    Motor_Stepping_Direction_POSITIVE = 0,

    // XY坐标负方向
    Motor_Stepping_Direction_NEGATIVE,
};


// ============================================================
// DM430数据
// ============================================================

struct Struct_Motor_Stepping_DM430_Data
{
    // 当前实际输出脉冲频率 Hz
    float Pulse_Frequency = 0.0f;

    // 当前直线速度 mm/s
    // 带正负号
    float Speed_mm_s = 0.0f;

    // 当前软件估计位置 mm
    float Position_mm = 0.0f;

    // 目标位置 mm
    float Target_Position_mm = 0.0f;

    // 当前方向
    Enum_Motor_Stepping_Direction Direction =
        Motor_Stepping_Direction_POSITIVE;
};


// ============================================================
// DM430
// ============================================================

class Class_Motor_Stepping_DM430
{
public:

    /**
     * @brief 初始化DM430
     *
     * @param htim                    基础定时器，例如 TIM13
     *
     * @param PUL_GPIO_Port           PUL GPIO端口
     * @param PUL_GPIO_Pin            PUL GPIO引脚
     *
     * @param DIR_GPIO_Port           DIR GPIO端口
     * @param DIR_GPIO_Pin            DIR GPIO引脚
     *
     * @param Pulse_Per_Round         驱动器每机械圈脉冲数
     *                                必须与SW5~SW8一致
     *
     * @param Travel_mm_Per_Round     电机转一圈，
     *                                XY轴实际移动多少mm
     *
     * @param Timer_Tick_Frequency    定时器计数频率
     *
     * @param Positive_DIR_GPIO_State XY正方向对应DIR GPIO电平
     *
     * @param PUL_Active_GPIO_State   PUL有效状态
     *
     * @param Max_Pulse_Frequency     最大允许脉冲频率
     */
    void Init(
        TIM_HandleTypeDef *htim,

        GPIO_TypeDef *PUL_GPIO_Port,
        uint16_t PUL_GPIO_Pin,

        GPIO_TypeDef *DIR_GPIO_Port,
        uint16_t DIR_GPIO_Pin,

        uint32_t Pulse_Per_Round,

        float Travel_mm_Per_Round,

        float Timer_Tick_Frequency =
            MOTOR_STEPPING_DM430_DEFAULT_TIMER_TICK_HZ,

        GPIO_PinState Positive_DIR_GPIO_State =
            GPIO_PIN_RESET,

        GPIO_PinState PUL_Active_GPIO_State =
            GPIO_PIN_RESET,

        float Max_Pulse_Frequency =
            MOTOR_STEPPING_DM430_DEFAULT_MAX_PULSE_FREQUENCY_HZ
    );


    // ========================================================
    // 原始脉冲控制
    //
    // pulse_count > 0：正方向
    // pulse_count < 0：负方向
    //
    // pulse_frequency：Hz
    // ========================================================

    HAL_StatusTypeDef Move_Pulses(
        int32_t pulse_count,
        float pulse_frequency
    );


    // ========================================================
    // 相对位移
    //
    // delta_mm > 0：正方向
    // delta_mm < 0：负方向
    //
    // speed_mm_s：速度绝对值
    // ========================================================

    HAL_StatusTypeDef Move_Relative_mm(
        float delta_mm,
        float speed_mm_s
    );


    // ========================================================
    // 绝对位置
    //
    // 必须先Set_Origin_mm()
    // ========================================================

    HAL_StatusTypeDef Move_To_mm(
        float target_position_mm,
        float speed_mm_s
    );


    // ========================================================
    // 连续速度运动
    //
    // speed > 0：正方向
    // speed < 0：负方向
    // speed = 0：停止
    // ========================================================

    HAL_StatusTypeDef Run_Speed_mm_s(
        float speed_mm_s
    );


    // ========================================================
    // 动态修改当前速度
    //
    // 给Class_Slope调用
    //
    // 对位置运动：
    //      只改变速度，不改变方向和剩余脉冲
    //
    // speed_mm_s传绝对值
    //
    // speed = 0：
    //      暂停，但保留Remaining_Pulses
    // ========================================================

    HAL_StatusTypeDef Set_Speed_mm_s(
        float speed_mm_s
    );


    // ========================================================
    // 暂停
    // 保留剩余脉冲
    // ========================================================

    void Pause();


    // ========================================================
    // 从Pause继续
    // ========================================================

    HAL_StatusTypeDef Resume();


    // ========================================================
    // 完全停止
    //
    // 清除当前运动任务
    // ========================================================

    void Stop();


    // ========================================================
    // 设置软件原点
    //
    // 通常回零开关触发后：
    //
    // Motor_X.Set_Origin_mm(0.0f);
    // ========================================================

    void Set_Origin_mm(
        float position_mm = 0.0f
    );


    // ========================================================
    // TIM中断回调
    // ========================================================

    void TIM_PeriodElapsedCallback(
        TIM_HandleTypeDef *htim
    );


    // ========================================================
    // Get
    // ========================================================

    inline Enum_Motor_Stepping_Status
    Get_Status() const
    {
        return Motor_Status;
    }


    inline Enum_Motor_Stepping_Control_Method
    Get_Control_Method() const
    {
        return Control_Method;
    }


    inline Enum_Motor_Stepping_Direction
    Get_Direction() const
    {
        return Data.Direction;
    }


    inline uint32_t
    Get_Pulse_Per_Round() const
    {
        return Pulse_Per_Round;
    }


    inline float
    Get_Travel_mm_Per_Round() const
    {
        return Travel_mm_Per_Round;
    }


    inline float
    Get_Pulse_Per_mm() const
    {
        return Pulse_Per_mm;
    }


    inline float
    Get_Pulse_Frequency() const
    {
        return Data.Pulse_Frequency;
    }


    inline float
    Get_Speed_mm_s() const
    {
        return Data.Speed_mm_s;
    }


    inline float
    Get_Position_mm() const
    {
        return Data.Position_mm;
    }


    inline float
    Get_Target_Position_mm() const
    {
        return Data.Target_Position_mm;
    }


    inline bool
    Get_Position_Valid() const
    {
        return Position_Valid;
    }


    inline int64_t
    Get_Position_Pulse() const
    {
        return Position_Pulse;
    }


    inline uint32_t
    Get_Remaining_Pulses() const
    {
        return Remaining_Pulses;
    }


protected:

    // ========================================================
    // 硬件
    // ========================================================

    TIM_HandleTypeDef *TIM_Handler = nullptr;


    GPIO_TypeDef *PUL_GPIO_Port = nullptr;

    uint16_t PUL_GPIO_Pin = 0;


    GPIO_TypeDef *DIR_GPIO_Port = nullptr;

    uint16_t DIR_GPIO_Pin = 0;


    // ========================================================
    // GPIO极性
    // ========================================================

    GPIO_PinState Positive_DIR_GPIO_State =
        GPIO_PIN_RESET;


    GPIO_PinState PUL_Active_GPIO_State =
        GPIO_PIN_RESET;


    // ========================================================
    // 机械参数
    // ========================================================

    // 一圈多少pulse
    uint32_t Pulse_Per_Round = 0U;


    // 电机转一圈对应直线移动多少mm
    float Travel_mm_Per_Round = 0.0f;


    // 每mm需要多少pulse
    float Pulse_Per_mm = 0.0f;


    // ========================================================
    // Timer参数
    // ========================================================

    float Timer_Tick_Frequency =
        MOTOR_STEPPING_DM430_DEFAULT_TIMER_TICK_HZ;


    float Max_Pulse_Frequency =
        MOTOR_STEPPING_DM430_DEFAULT_MAX_PULSE_FREQUENCY_HZ;


    // 当前完整PUL周期tick数
    volatile uint32_t Pulse_Period_Ticks = 0U;


    uint32_t Pulse_Active_Ticks = 0U;


    uint32_t Pulse_Inactive_Min_Ticks = 0U;


    // ========================================================
    // Pulse状态机
    // ========================================================

    enum Enum_Pulse_Phase : uint8_t
    {
        Pulse_Phase_IDLE = 0,

        Pulse_Phase_ACTIVE,

        Pulse_Phase_INACTIVE,
    };


    volatile Enum_Pulse_Phase Pulse_Phase =
        Pulse_Phase_IDLE;


    // ========================================================
    // 多圈位置
    // ========================================================

    // 最核心的位置变量
    //
    // int64_t支持大量多圈累计
    volatile int64_t Position_Pulse = 0;


    bool Position_Valid = false;


    // 当前有限运动还剩多少pulse
    volatile uint32_t Remaining_Pulses = 0U;


    // 当前方向符号
    // +1 正向
    // -1 负向
    int8_t Direction_Sign = 1;


    // ========================================================
    // 状态
    // ========================================================

    Enum_Motor_Stepping_Status Motor_Status =
        Motor_Stepping_Status_STOP;


    Enum_Motor_Stepping_Control_Method Control_Method =
        Motor_Stepping_Control_Method_STOP;


    Struct_Motor_Stepping_DM430_Data Data;


    // ========================================================
    // Pause前的速度
    // ========================================================

    float Resume_Speed_mm_s = 0.0f;


    // ========================================================
    // 内部方法
    // ========================================================

    HAL_StatusTypeDef Start_Finite_Pulses(
        uint32_t pulse_num,

        Enum_Motor_Stepping_Direction direction,

        float pulse_frequency,

        Enum_Motor_Stepping_Control_Method method
    );


    HAL_StatusTypeDef Configure_Pulse_Frequency(
        float pulse_frequency
    );


    HAL_StatusTypeDef Start_Timer_Output();


    void Stop_Timer_Only();


    void Set_Direction(
        Enum_Motor_Stepping_Direction direction
    );


    void Set_Pulse_Active(
        bool active
    );


    void Finish_Motion();


    void Add_One_Pulse_To_Position();


    void Refresh_Position_mm();


    float Speed_mm_s_To_Pulse_Frequency(
        float speed_mm_s
    ) const;


    float Pulse_Frequency_To_Speed_mm_s(
        float pulse_frequency
    ) const;


    int64_t Position_mm_To_Pulse(
        float position_mm
    ) const;


    static GPIO_PinState Invert_GPIO_State(
        GPIO_PinState state
    );


    static void Delay_Us(
        uint32_t us
    );
};


#endif