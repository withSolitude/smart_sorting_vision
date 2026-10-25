/**
 * @file dvc_motor_stepping.cpp
 * @brief DM430步进电机直线轴驱动
 */

#include "2_Device/Motor/Motor_Stepping/dvc_motor_stepping.h"

#include <cmath>
#include <climits>


// ============================================================
// Init
// ============================================================

void Class_Motor_Stepping_DM430::Init(
    TIM_HandleTypeDef *htim,

    GPIO_TypeDef *__PUL_GPIO_Port,
    uint16_t __PUL_GPIO_Pin,

    GPIO_TypeDef *__DIR_GPIO_Port,
    uint16_t __DIR_GPIO_Pin,

    uint32_t __Pulse_Per_Round,

    float __Travel_mm_Per_Round,

    float __Timer_Tick_Frequency,

    GPIO_PinState __Positive_DIR_GPIO_State,

    GPIO_PinState __PUL_Active_GPIO_State,

    float __Max_Pulse_Frequency)
{
    // ========================================================
    // Hardware
    // ========================================================

    TIM_Handler =
        htim;


    PUL_GPIO_Port =
        __PUL_GPIO_Port;

    PUL_GPIO_Pin =
        __PUL_GPIO_Pin;


    DIR_GPIO_Port =
        __DIR_GPIO_Port;

    DIR_GPIO_Pin =
        __DIR_GPIO_Pin;


    // ========================================================
    // Mechanical parameters
    // ========================================================

    Pulse_Per_Round =
        __Pulse_Per_Round;


    Travel_mm_Per_Round =
        __Travel_mm_Per_Round;


    if (Pulse_Per_Round > 0U &&
        Travel_mm_Per_Round > 0.0f)
    {
        Pulse_Per_mm =
            static_cast<float>(
                Pulse_Per_Round
            ) /
            Travel_mm_Per_Round;
    }
    else
    {
        Pulse_Per_mm =
            0.0f;
    }


    // ========================================================
    // Timer
    // ========================================================

    Timer_Tick_Frequency =
        (__Timer_Tick_Frequency > 0.0f)
            ? __Timer_Tick_Frequency
            : MOTOR_STEPPING_DM430_DEFAULT_TIMER_TICK_HZ;


    Max_Pulse_Frequency =
        (__Max_Pulse_Frequency > 0.0f)
            ? __Max_Pulse_Frequency
            : MOTOR_STEPPING_DM430_DEFAULT_MAX_PULSE_FREQUENCY_HZ;


    // ========================================================
    // GPIO polarity
    // ========================================================

    Positive_DIR_GPIO_State =
        __Positive_DIR_GPIO_State;


    PUL_Active_GPIO_State =
        __PUL_Active_GPIO_State;


    // ========================================================
    // Pulse width ticks
    // ========================================================

    Pulse_Active_Ticks =
        static_cast<uint32_t>(
            std::ceil(
                Timer_Tick_Frequency *
                static_cast<float>(
                    MOTOR_STEPPING_DM430_PULSE_ACTIVE_TIME_US
                ) /
                1000000.0f
            )
        );


    if (Pulse_Active_Ticks < 1U)
    {
        Pulse_Active_Ticks = 1U;
    }


    Pulse_Inactive_Min_Ticks =
        static_cast<uint32_t>(
            std::ceil(
                Timer_Tick_Frequency *
                static_cast<float>(
                    MOTOR_STEPPING_DM430_PULSE_INACTIVE_MIN_TIME_US
                ) /
                1000000.0f
            )
        );


    if (Pulse_Inactive_Min_Ticks < 1U)
    {
        Pulse_Inactive_Min_Ticks = 1U;
    }


    // ========================================================
    // Reset states
    // ========================================================

    Position_Pulse =
        0;


    Position_Valid =
        false;


    Remaining_Pulses =
        0U;


    Direction_Sign =
        1;


    Pulse_Phase =
        Pulse_Phase_IDLE;


    Motor_Status =
        Motor_Stepping_Status_STOP;


    Control_Method =
        Motor_Stepping_Control_Method_STOP;


    Data =
        Struct_Motor_Stepping_DM430_Data{};


    Resume_Speed_mm_s =
        0.0f;


    // ========================================================
    // Timer reset
    // ========================================================

    if (TIM_Handler != nullptr)
    {
        HAL_TIM_Base_Stop_IT(
            TIM_Handler
        );


        __HAL_TIM_SET_COUNTER(
            TIM_Handler,
            0U
        );


        // ARR立即生效
        CLEAR_BIT(
            TIM_Handler->Instance->CR1,
            TIM_CR1_ARPE
        );


        __HAL_TIM_CLEAR_FLAG(
            TIM_Handler,
            TIM_FLAG_UPDATE
        );
    }


    // ========================================================
    // GPIO safe state
    // ========================================================

    Set_Pulse_Active(false);


    Set_Direction(
        Motor_Stepping_Direction_POSITIVE
    );


    // ========================================================
    // DWT
    // ========================================================

    CoreDebug->DEMCR |=
        CoreDebug_DEMCR_TRCENA_Msk;


    DWT->CYCCNT =
        0U;


    DWT->CTRL |=
        DWT_CTRL_CYCCNTENA_Msk;
}


// ============================================================
// Move_Pulses
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Move_Pulses(
    int32_t pulse_count,
    float pulse_frequency)
{
    if (TIM_Handler == nullptr ||
        pulse_count == 0 ||
        pulse_frequency <= 0.0f)
    {
        return HAL_ERROR;
    }


    const int64_t count =
        static_cast<int64_t>(
            pulse_count
        );


    const uint64_t pulse_num =
        (count > 0)
            ? static_cast<uint64_t>(count)
            : static_cast<uint64_t>(-count);


    if (pulse_num >
        static_cast<uint64_t>(
            UINT32_MAX
        ))
    {
        return HAL_ERROR;
    }


    const Enum_Motor_Stepping_Direction direction =
        (pulse_count > 0)
            ? Motor_Stepping_Direction_POSITIVE
            : Motor_Stepping_Direction_NEGATIVE;


    return Start_Finite_Pulses(
        static_cast<uint32_t>(
            pulse_num
        ),

        direction,

        pulse_frequency,

        Motor_Stepping_Control_Method_PULSES
    );
}


// ============================================================
// Move_Relative_mm
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Move_Relative_mm(
    float delta_mm,
    float speed_mm_s)
{
    if (Pulse_Per_mm <= 0.0f ||
        speed_mm_s <= 0.0f)
    {
        return HAL_ERROR;
    }


    if (std::fabs(delta_mm) <
        0.000001f)
    {
        return HAL_OK;
    }


    const double pulse_double =
        static_cast<double>(
            std::fabs(delta_mm)
        ) *
        static_cast<double>(
            Pulse_Per_mm
        );


    const uint64_t pulse_num =
        static_cast<uint64_t>(
            std::llround(
                pulse_double
            )
        );


    if (pulse_num == 0U)
    {
        return HAL_OK;
    }


    if (pulse_num >
        static_cast<uint64_t>(
            UINT32_MAX
        ))
    {
        return HAL_ERROR;
    }


    const Enum_Motor_Stepping_Direction direction =
        (delta_mm > 0.0f)
            ? Motor_Stepping_Direction_POSITIVE
            : Motor_Stepping_Direction_NEGATIVE;


    const float pulse_frequency =
        Speed_mm_s_To_Pulse_Frequency(
            std::fabs(speed_mm_s)
        );


    // 如果已经建立坐标
    // 同时记录目标位置
    if (Position_Valid)
    {
        Data.Target_Position_mm =
            Data.Position_mm +
            delta_mm;
    }


    return Start_Finite_Pulses(
        static_cast<uint32_t>(
            pulse_num
        ),

        direction,

        pulse_frequency,

        Motor_Stepping_Control_Method_RELATIVE_MM
    );
}


// ============================================================
// Move_To_mm
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Move_To_mm(
    float target_position_mm,
    float speed_mm_s)
{
    if (!Position_Valid ||
        Pulse_Per_mm <= 0.0f ||
        speed_mm_s <= 0.0f)
    {
        return HAL_ERROR;
    }


    const int64_t target_pulse =
        Position_mm_To_Pulse(
            target_position_mm
        );


    const int64_t delta_pulse =
        target_pulse -
        Position_Pulse;


    Data.Target_Position_mm =
        target_position_mm;


    if (delta_pulse == 0)
    {
        return HAL_OK;
    }


    const uint64_t pulse_num =
        (delta_pulse > 0)
            ? static_cast<uint64_t>(
                  delta_pulse
              )
            : static_cast<uint64_t>(
                  -delta_pulse
              );


    if (pulse_num >
        static_cast<uint64_t>(
            UINT32_MAX
        ))
    {
        return HAL_ERROR;
    }


    const Enum_Motor_Stepping_Direction direction =
        (delta_pulse > 0)
            ? Motor_Stepping_Direction_POSITIVE
            : Motor_Stepping_Direction_NEGATIVE;


    const float pulse_frequency =
        Speed_mm_s_To_Pulse_Frequency(
            std::fabs(speed_mm_s)
        );


    return Start_Finite_Pulses(
        static_cast<uint32_t>(
            pulse_num
        ),

        direction,

        pulse_frequency,

        Motor_Stepping_Control_Method_ABSOLUTE_MM
    );
}


// ============================================================
// Continuous speed
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Run_Speed_mm_s(
    float speed_mm_s)
{
    if (TIM_Handler == nullptr ||
        Pulse_Per_mm <= 0.0f)
    {
        return HAL_ERROR;
    }


    if (std::fabs(speed_mm_s) <
        0.000001f)
    {
        Stop();

        return HAL_OK;
    }


    Stop();


    const Enum_Motor_Stepping_Direction direction =
        (speed_mm_s > 0.0f)
            ? Motor_Stepping_Direction_POSITIVE
            : Motor_Stepping_Direction_NEGATIVE;


    const float speed_abs =
        std::fabs(
            speed_mm_s
        );


    const float pulse_frequency =
        Speed_mm_s_To_Pulse_Frequency(
            speed_abs
        );


    if (Configure_Pulse_Frequency(
            pulse_frequency
        ) != HAL_OK)
    {
        return HAL_ERROR;
    }


    Set_Direction(
        direction
    );


    Delay_Us(
        MOTOR_STEPPING_DM430_DIRECTION_SETUP_TIME_US
    );


    Direction_Sign =
        (direction ==
         Motor_Stepping_Direction_POSITIVE)
            ? 1
            : -1;


    Data.Direction =
        direction;


    Data.Speed_mm_s =
        static_cast<float>(
            Direction_Sign
        ) *
        Pulse_Frequency_To_Speed_mm_s(
            Data.Pulse_Frequency
        );


    Resume_Speed_mm_s =
        std::fabs(
            Data.Speed_mm_s
        );


    Remaining_Pulses =
        0U;


    Control_Method =
        Motor_Stepping_Control_Method_SPEED_MM_S;


    Motor_Status =
        Motor_Stepping_Status_RUNNING;


    const HAL_StatusTypeDef status =
        Start_Timer_Output();


    if (status != HAL_OK)
    {
        Stop();

        return status;
    }


    return HAL_OK;
}


// ============================================================
// Dynamic speed update
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Set_Speed_mm_s(
    float speed_mm_s)
{
    if (Control_Method ==
            Motor_Stepping_Control_Method_STOP ||
        Pulse_Per_mm <= 0.0f)
    {
        return HAL_ERROR;
    }


    const float speed_abs =
        std::fabs(
            speed_mm_s
        );


    // ========================================================
    // 0速度 = 暂停
    // ========================================================

    if (speed_abs <
        0.000001f)
    {
        Pause();

        return HAL_OK;
    }


    const float pulse_frequency =
        Speed_mm_s_To_Pulse_Frequency(
            speed_abs
        );


    if (Configure_Pulse_Frequency(
            pulse_frequency
        ) != HAL_OK)
    {
        return HAL_ERROR;
    }


    Resume_Speed_mm_s =
        speed_abs;


    Data.Speed_mm_s =
        static_cast<float>(
            Direction_Sign
        ) *
        Pulse_Frequency_To_Speed_mm_s(
            Data.Pulse_Frequency
        );


    // 如果原来处于暂停
    // 自动重新启动
    if (Motor_Status ==
        Motor_Stepping_Status_PAUSED)
    {
        Motor_Status =
            Motor_Stepping_Status_RUNNING;


        return Start_Timer_Output();
    }


    /*
     * RUNNING状态下：
     *
     * 不Stop
     * 不清Remaining_Pulses
     * 不重新设置方向
     *
     * 只修改Pulse_Period_Ticks。
     *
     * TIM中断会在下一个脉冲周期自动使用新周期。
     */

    return HAL_OK;
}


// ============================================================
// Pause
// ============================================================

void Class_Motor_Stepping_DM430::Pause()
{
    if (Motor_Status !=
        Motor_Stepping_Status_RUNNING)
    {
        return;
    }


    if (std::fabs(
            Data.Speed_mm_s
        ) > 0.0f)
    {
        Resume_Speed_mm_s =
            std::fabs(
                Data.Speed_mm_s
            );
    }


    Stop_Timer_Only();


    Data.Pulse_Frequency =
        0.0f;


    Data.Speed_mm_s =
        0.0f;


    Motor_Status =
        Motor_Stepping_Status_PAUSED;
}


// ============================================================
// Resume
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Resume()
{
    if (Motor_Status !=
            Motor_Stepping_Status_PAUSED ||
        Resume_Speed_mm_s <= 0.0f)
    {
        return HAL_ERROR;
    }


    const float pulse_frequency =
        Speed_mm_s_To_Pulse_Frequency(
            Resume_Speed_mm_s
        );


    if (Configure_Pulse_Frequency(
            pulse_frequency
        ) != HAL_OK)
    {
        return HAL_ERROR;
    }


    Data.Speed_mm_s =
        static_cast<float>(
            Direction_Sign
        ) *
        Pulse_Frequency_To_Speed_mm_s(
            Data.Pulse_Frequency
        );


    Motor_Status =
        Motor_Stepping_Status_RUNNING;


    return Start_Timer_Output();
}


// ============================================================
// Stop
// ============================================================

void Class_Motor_Stepping_DM430::Stop()
{
    Stop_Timer_Only();


    Remaining_Pulses =
        0U;


    Pulse_Phase =
        Pulse_Phase_IDLE;


    Data.Pulse_Frequency =
        0.0f;


    Data.Speed_mm_s =
        0.0f;


    Resume_Speed_mm_s =
        0.0f;


    Motor_Status =
        Motor_Stepping_Status_STOP;


    Control_Method =
        Motor_Stepping_Control_Method_STOP;


    Refresh_Position_mm();
}


// ============================================================
// Set Origin
// ============================================================

void Class_Motor_Stepping_DM430::Set_Origin_mm(
    float position_mm)
{
    if (Pulse_Per_mm <= 0.0f)
    {
        Position_Valid =
            false;

        return;
    }


    Position_Pulse =
        Position_mm_To_Pulse(
            position_mm
        );


    Position_Valid =
        true;


    Refresh_Position_mm();


    Data.Target_Position_mm =
        Data.Position_mm;
}


// ============================================================
// Start finite pulses
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Start_Finite_Pulses(
    uint32_t pulse_num,

    Enum_Motor_Stepping_Direction direction,

    float pulse_frequency,

    Enum_Motor_Stepping_Control_Method method)
{
    if (TIM_Handler == nullptr ||
        pulse_num == 0U ||
        pulse_frequency <= 0.0f)
    {
        return HAL_ERROR;
    }


    Stop();


    if (Configure_Pulse_Frequency(
            pulse_frequency
        ) != HAL_OK)
    {
        return HAL_ERROR;
    }


    Set_Direction(
        direction
    );


    // DIR -> PUL setup time
    Delay_Us(
        MOTOR_STEPPING_DM430_DIRECTION_SETUP_TIME_US
    );


    Direction_Sign =
        (direction ==
         Motor_Stepping_Direction_POSITIVE)
            ? 1
            : -1;


    Data.Direction =
        direction;


    Data.Speed_mm_s =
        static_cast<float>(
            Direction_Sign
        ) *
        Pulse_Frequency_To_Speed_mm_s(
            Data.Pulse_Frequency
        );


    Resume_Speed_mm_s =
        std::fabs(
            Data.Speed_mm_s
        );


    Remaining_Pulses =
        pulse_num;


    Control_Method =
        method;


    Motor_Status =
        Motor_Stepping_Status_RUNNING;


    const HAL_StatusTypeDef status =
        Start_Timer_Output();


    if (status != HAL_OK)
    {
        Stop();

        return status;
    }


    return HAL_OK;
}


// ============================================================
// Frequency configuration
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Configure_Pulse_Frequency(
    float pulse_frequency)
{
    if (TIM_Handler == nullptr ||
        Timer_Tick_Frequency <= 0.0f ||
        pulse_frequency <= 0.0f)
    {
        return HAL_ERROR;
    }


    // ========================================================
    // 最大频率限制
    // ========================================================

    if (pulse_frequency >
        Max_Pulse_Frequency)
    {
        pulse_frequency =
            Max_Pulse_Frequency;
    }


    const uint64_t period_ticks =
        static_cast<uint64_t>(
            std::llround(
                static_cast<double>(
                    Timer_Tick_Frequency
                ) /
                static_cast<double>(
                    pulse_frequency
                )
            )
        );


    if (period_ticks <=
        static_cast<uint64_t>(
            Pulse_Active_Ticks +
            Pulse_Inactive_Min_Ticks
        ))
    {
        return HAL_ERROR;
    }


    const uint64_t inactive_ticks =
        period_ticks -
        Pulse_Active_Ticks;


    const uint64_t timer_arr_max =
        (IS_TIM_32B_COUNTER_INSTANCE(
             TIM_Handler->Instance) != 0U)
            ? static_cast<uint64_t>(
                  UINT32_MAX
              )
            : static_cast<uint64_t>(
                  UINT16_MAX
              );


    /*
     * ACTIVE和INACTIVE分别单独计时，
     * 所以两个阶段都必须能装入ARR。
     */

    if ((Pulse_Active_Ticks - 1U) >
            timer_arr_max ||
        (inactive_ticks - 1U) >
            timer_arr_max)
    {
        return HAL_ERROR;
    }


    /*
     * 最后赋值。
     *
     * Cortex-M7上32bit写入是原子的，
     * TIM中断下一周期即可使用。
     */
    Pulse_Period_Ticks =
        static_cast<uint32_t>(
            period_ticks
        );


    Data.Pulse_Frequency =
        Timer_Tick_Frequency /
        static_cast<float>(
            Pulse_Period_Ticks
        );


    return HAL_OK;
}


// ============================================================
// Start timer output
// ============================================================

HAL_StatusTypeDef
Class_Motor_Stepping_DM430::Start_Timer_Output()
{
    if (TIM_Handler == nullptr ||
        PUL_GPIO_Port == nullptr ||
        PUL_GPIO_Pin == 0U ||
        Pulse_Period_Ticks == 0U)
    {
        return HAL_ERROR;
    }


    HAL_TIM_Base_Stop_IT(
        TIM_Handler
    );


    // ========================================================
    // 第一发PUL立即开始
    // ========================================================

    Set_Pulse_Active(true);


    Pulse_Phase =
        Pulse_Phase_ACTIVE;


    // ACTIVE持续5us
    __HAL_TIM_SET_AUTORELOAD(
        TIM_Handler,
        Pulse_Active_Ticks - 1U
    );


    __HAL_TIM_SET_COUNTER(
        TIM_Handler,
        0U
    );


    __HAL_TIM_CLEAR_FLAG(
        TIM_Handler,
        TIM_FLAG_UPDATE
    );


    return HAL_TIM_Base_Start_IT(
        TIM_Handler
    );
}


// ============================================================
// TIM callback
// ============================================================

void
Class_Motor_Stepping_DM430::TIM_PeriodElapsedCallback(
    TIM_HandleTypeDef *htim)
{
    if (htim == nullptr ||
        TIM_Handler == nullptr ||
        htim->Instance !=
            TIM_Handler->Instance ||
        Motor_Status !=
            Motor_Stepping_Status_RUNNING)
    {
        return;
    }


    // ========================================================
    // ACTIVE结束
    //
    // 完成一个有效脉冲
    // ========================================================

    if (Pulse_Phase ==
        Pulse_Phase_ACTIVE)
    {
        // PUL恢复非有效
        Set_Pulse_Active(false);


        // ====================================================
        // 这个PUL已经真正输出完成
        // 在这里更新位置
        // ====================================================

        Add_One_Pulse_To_Position();


        // ====================================================
        // 有限运动
        // ====================================================

        if (Control_Method !=
            Motor_Stepping_Control_Method_SPEED_MM_S)
        {
            if (Remaining_Pulses > 0U)
            {
                Remaining_Pulses--;
            }


            if (Remaining_Pulses == 0U)
            {
                Finish_Motion();

                return;
            }
        }


        // ====================================================
        // 进入INACTIVE阶段
        //
        // 使用最新Pulse_Period_Ticks
        // 所以外部Slope修改速度后自动生效
        // ====================================================

        uint32_t inactive_ticks =
            Pulse_Period_Ticks -
            Pulse_Active_Ticks;


        if (inactive_ticks <
            Pulse_Inactive_Min_Ticks)
        {
            inactive_ticks =
                Pulse_Inactive_Min_Ticks;
        }


        __HAL_TIM_SET_AUTORELOAD(
            TIM_Handler,
            inactive_ticks - 1U
        );


        __HAL_TIM_SET_COUNTER(
            TIM_Handler,
            0U
        );


        Pulse_Phase =
            Pulse_Phase_INACTIVE;


        return;
    }


    // ========================================================
    // INACTIVE结束
    //
    // 开始下一个PUL
    // ========================================================

    if (Pulse_Phase ==
        Pulse_Phase_INACTIVE)
    {
        Set_Pulse_Active(true);


        __HAL_TIM_SET_AUTORELOAD(
            TIM_Handler,
            Pulse_Active_Ticks - 1U
        );


        __HAL_TIM_SET_COUNTER(
            TIM_Handler,
            0U
        );


        Pulse_Phase =
            Pulse_Phase_ACTIVE;


        return;
    }
}


// ============================================================
// Finish motion
// ============================================================

void
Class_Motor_Stepping_DM430::Finish_Motion()
{
    Stop_Timer_Only();


    Remaining_Pulses =
        0U;


    Pulse_Phase =
        Pulse_Phase_IDLE;


    Data.Pulse_Frequency =
        0.0f;


    Data.Speed_mm_s =
        0.0f;


    Resume_Speed_mm_s =
        0.0f;


    Motor_Status =
        Motor_Stepping_Status_STOP;


    Control_Method =
        Motor_Stepping_Control_Method_STOP;


    Refresh_Position_mm();
}


// ============================================================
// Stop timer only
// ============================================================

void
Class_Motor_Stepping_DM430::Stop_Timer_Only()
{
    if (TIM_Handler != nullptr)
    {
        HAL_TIM_Base_Stop_IT(
            TIM_Handler
        );


        __HAL_TIM_SET_COUNTER(
            TIM_Handler,
            0U
        );


        __HAL_TIM_CLEAR_FLAG(
            TIM_Handler,
            TIM_FLAG_UPDATE
        );
    }


    Set_Pulse_Active(false);


    Pulse_Phase =
        Pulse_Phase_IDLE;
}


// ============================================================
// Direction
// ============================================================

void
Class_Motor_Stepping_DM430::Set_Direction(
    Enum_Motor_Stepping_Direction direction)
{
    Data.Direction =
        direction;


    if (DIR_GPIO_Port == nullptr ||
        DIR_GPIO_Pin == 0U)
    {
        return;
    }


    GPIO_PinState state;


    if (direction ==
        Motor_Stepping_Direction_POSITIVE)
    {
        state =
            Positive_DIR_GPIO_State;
    }
    else
    {
        state =
            Invert_GPIO_State(
                Positive_DIR_GPIO_State
            );
    }


    HAL_GPIO_WritePin(
        DIR_GPIO_Port,
        DIR_GPIO_Pin,
        state
    );
}


// ============================================================
// Pulse GPIO
// ============================================================

void
Class_Motor_Stepping_DM430::Set_Pulse_Active(
    bool active)
{
    if (PUL_GPIO_Port == nullptr ||
        PUL_GPIO_Pin == 0U)
    {
        return;
    }


    GPIO_PinState state;


    if (active)
    {
        state =
            PUL_Active_GPIO_State;
    }
    else
    {
        state =
            Invert_GPIO_State(
                PUL_Active_GPIO_State
            );
    }


    HAL_GPIO_WritePin(
        PUL_GPIO_Port,
        PUL_GPIO_Pin,
        state
    );
}


// ============================================================
// Add one pulse
// ============================================================

void
Class_Motor_Stepping_DM430::Add_One_Pulse_To_Position()
{
    Position_Pulse +=
        static_cast<int64_t>(
            Direction_Sign
        );


    if (Position_Valid)
    {
        Refresh_Position_mm();
    }
}


// ============================================================
// Refresh position
// ============================================================

void
Class_Motor_Stepping_DM430::Refresh_Position_mm()
{
    if (!Position_Valid ||
        Pulse_Per_mm <= 0.0f)
    {
        return;
    }


    Data.Position_mm =
        static_cast<float>(
            Position_Pulse
        ) /
        Pulse_Per_mm;
}


// ============================================================
// mm/s -> Hz
// ============================================================

float
Class_Motor_Stepping_DM430::Speed_mm_s_To_Pulse_Frequency(
    float speed_mm_s) const
{
    if (Pulse_Per_mm <= 0.0f)
    {
        return 0.0f;
    }


    return std::fabs(
               speed_mm_s
           ) *
           Pulse_Per_mm;
}


// ============================================================
// Hz -> mm/s
// ============================================================

float
Class_Motor_Stepping_DM430::Pulse_Frequency_To_Speed_mm_s(
    float pulse_frequency) const
{
    if (Pulse_Per_mm <= 0.0f)
    {
        return 0.0f;
    }


    return pulse_frequency /
           Pulse_Per_mm;
}


// ============================================================
// mm -> pulse
// ============================================================

int64_t
Class_Motor_Stepping_DM430::Position_mm_To_Pulse(
    float position_mm) const
{
    if (Pulse_Per_mm <= 0.0f)
    {
        return 0;
    }


    return static_cast<int64_t>(
        std::llround(
            static_cast<double>(
                position_mm
            ) *
            static_cast<double>(
                Pulse_Per_mm
            )
        )
    );
}


// ============================================================
// GPIO invert
// ============================================================

GPIO_PinState
Class_Motor_Stepping_DM430::Invert_GPIO_State(
    GPIO_PinState state)
{
    return
        (state == GPIO_PIN_SET)
            ? GPIO_PIN_RESET
            : GPIO_PIN_SET;
}


// ============================================================
// DWT delay
// ============================================================

void
Class_Motor_Stepping_DM430::Delay_Us(
    uint32_t us)
{
    if (us == 0U)
    {
        return;
    }


    CoreDebug->DEMCR |=
        CoreDebug_DEMCR_TRCENA_Msk;


    if ((DWT->CTRL &
         DWT_CTRL_CYCCNTENA_Msk) == 0U)
    {
        DWT->CYCCNT =
            0U;


        DWT->CTRL |=
            DWT_CTRL_CYCCNTENA_Msk;
    }


    const uint32_t cycles_per_us =
        SystemCoreClock /
        1000000U;


    const uint32_t start =
        DWT->CYCCNT;


    const uint32_t target_cycles =
        cycles_per_us *
        us;


    while (
        static_cast<uint32_t>(
            DWT->CYCCNT -
            start
        ) <
        target_cycles
    )
    {
        __NOP();
    }
}