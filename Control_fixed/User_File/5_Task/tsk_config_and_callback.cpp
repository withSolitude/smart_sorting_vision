//
// Created by Lenovo on 2026/1/12.
//
#include "5_Task/tsk_config_and_callback.h"
#include "4_Interaction/ita_robot.h"
#include "2_Device/Vofa/dvc_vofa.h"
#include "2_Device/Vofa/dvc_Vofa_Tuner.h"
#include "2_Device/Motor/Servo_Motor_UART/dvc_servo_motor_uart.h"
#include "3_Robot/xy_motion_config.h"
#include "2_Device/TOF050F/dvc_tof050f.h"
//调试专用

volatile HAL_StatusTypeDef DM_Test_Tx_Status = HAL_ERROR;

volatile uint32_t DM_Test_Tx_Free_Level = 0;

volatile uint32_t DM_Test_Rx_Fill_Level = 0;

volatile uint32_t DM_Test_Rx_ID = 0xFFFFFFFF;

volatile uint8_t DM_Test_Rx_Data[8] =
{
    0, 0, 0, 0,
    0, 0, 0, 0
};

volatile uint32_t DM_Test_ErrorCode = 0;

volatile uint32_t DM_Test_LastErrorCode = 0;

volatile uint32_t DM_Test_BusOff = 0;

volatile uint32_t DM_Test_TxErrorCnt = 0;

volatile uint32_t DM_Test_RxErrorCnt = 0;


uint8_t a =0;
uint8_t b =0;
uint8_t c =0;
uint8_t d =0;

bool volatile init_finished = false;
uint32_t flag = 0;
float servo_angle = 0;

// 这就是我们的牢雄！！！！！
Class_Robot robot;
// Class_Vofa_Tuner Vofa_Tuner;
// Class_Vofa_JustFloat Vofa_JustFloat;
Class_Servo_Motor Servo_Motor;

// 调试器/人工原点入口：先写入 X/Y，再在停机状态下将 Request 置 1。
// Task_Loop 遇 HAL_BUSY 会持续重试，直到空闲且反馈新鲜；HAL_OK/HAL_ERROR 后清请求。
// 只有把滑台放到已知原点后才能发起请求。
// 该入口不会改变 PC1 实体按键语义。
volatile uint8_t XY_Origin_Confirm_Request = 0U;
volatile float XY_Origin_Confirm_X_mm = 0.0f;
volatile float XY_Origin_Confirm_Y_mm = 0.0f;
volatile HAL_StatusTypeDef XY_Origin_Confirm_Status = HAL_ERROR;
volatile uint8_t XY_Collision_Homing_Request = 0U;
volatile HAL_StatusTypeDef XY_Collision_Homing_Status = HAL_ERROR;

uint16_t tof_distance =
    0;

Class_Motor_DM_Normal DM_Motor;


bool tof_online =
    false;


/**
 *@brief VOFA+ 串口绘图UART1初始化
 *
 */
void Vofa_JustFloat_Init(void)
{
    // Vofa_JustFloat.Init(&huart7,8,20);
    //绑定通道
    // Vofa_JustFloat.Bind_Channel(0,&tar_angle);
}

/**
 *@brief VOFA+ 串口绘图slider初始化
 *
 */
void Vofa_Tuner_Init()
{
    // Vofa_Tuner.Init(&huart7,64);

    // Vofa_Tuner.Set_Reset_Integral_On_Tune(1);
    //绑定
    // Vofa_Tuner.Bind_PID(1000,&robot.Shooter.Motor_Driver.PID_Angle);

}

/**
 *@brief UART1 VOFA+回调函数
 *
 */
void Vofa_JustFloat_UART1_Callback()
{
    // a = robot.AutoAim.Get_Rx_Mode();
    // b = robot.AutoAim.Get_Target_Pitch();
    // c = robot.AutoAim.Get_Target_Yaw();
}

/**
 *@brief UART1 VOFA+调参回调函数
 *
 *@param Buffer UART1收到的消息
 *@param Length 长度
 */
void Vofa_Tuner_UART1_Callback(uint8_t *Buffer, uint16_t Length)
{
    // Vofa_Tuner.UART_RxCpltCallback(Buffer, Length);
}

/**
 *
 * @param Buffer UART3 舵机
 * @param Length 长度
 */
void Servo_UART3_Callback(uint8_t *Buffer, uint16_t Length)
{
    robot.Holding_Jaw.Yaw_Servo_Motor.UART_RxCpltCallback(Buffer, Length);
    robot.Holding_Jaw.Lift_Servo_Motor.UART_RxCpltCallback(Buffer, Length);
    robot.Holding_Jaw.Claw_Servo_Motor.UART_RxCpltCallback(Buffer, Length);
}

/**
 *
 * @param Buffer UART4 舵机
 * @param Length 长度
 */
void Servo_UART4_Callback(uint8_t *Buffer, uint16_t Length)
{
    robot.Sorting_Storage.UART_RxCpltCallback(&huart4, Buffer, Length);
    // Servo_Motor.UART_RxCpltCallback(Buffer, Length);

}

/**
 *
 * @param Buffer UART2 激光测距
 * @param Length 长度
 */
void TOF050F_UART2_Callback(uint8_t *Buffer,uint16_t Length)
{
    robot.Holding_Jaw.TOF050F.UART_RxCpltCallback(Buffer,Length);
}

/**
 * @brief 每3600s调用一次
 *
 */
void Task3600s_Callback()
{
    SYS_Timestamp.TIM_3600s_PeriodElapsedCallback();
}


/**
 *@brief CAN1回调函数
 *
 *@param CAN_RxMessage CAN1收到的消息
 */
void Device_CAN1_Callback(FDCAN_RxHeaderTypeDef &Header, uint8_t *Buffer)
{
    if (Header.IdType == FDCAN_EXTENDED_ID)
    {
        if (init_finished && XY_Config::CAN_Bus() == &hfdcan1)
        {
            robot.Holding_Jaw.XY_Motion.CAN_RxCpltCallback(Header, Buffer);
        }
        return;
    }
    // DM4310 使用标准帧，Emm 地址 1 使用扩展帧；即使数值相同，
    // 也必须按帧类型分流，不能把 Emm 应答误解析成 DM4310 反馈。
    if (init_finished && Header.IdType == FDCAN_STANDARD_ID)
    {
        robot.Sorting_Storage.CAN_RxCpltCallback(&hfdcan1, Header);
    }
}

void Device_CAN2_Callback(FDCAN_RxHeaderTypeDef &Header, uint8_t *Buffer)
{
    if (init_finished && XY_Config::CAN_Bus() == &hfdcan2 && Header.IdType == FDCAN_EXTENDED_ID)
    {
        robot.Holding_Jaw.XY_Motion.CAN_RxCpltCallback(Header, Buffer);
    }
}

void Device_CAN3_Callback(FDCAN_RxHeaderTypeDef &Header, uint8_t *Buffer)
{
    if (init_finished && XY_Config::CAN_Bus() == &hfdcan3 && Header.IdType == FDCAN_EXTENDED_ID)
    {
        robot.Holding_Jaw.XY_Motion.CAN_RxCpltCallback(Header, Buffer);
    }
}


/**
 *@brief 虚拟串口上位机接收回调
 *
 */
void AutoAim_USB_Callback(uint8_t *Buffer, uint16_t Length)
{
    // if (!init_finished)
    // {
    //     return;
    // }
    //
    // // 达妙云台还没反馈正常前，不处理视觉数据
    // // 避免上电阶段 USB 高频数据影响 CAN / DM 使能过程
    // if (robot.Gimbal.Yaw_Motor_MIT.Get_Status() != Motor_DM_Status_ENABLE ||
    //     robot.Gimbal.Pitch_Motor_MIT.Get_Status() != Motor_DM_Status_ENABLE)
    // {
    //     return;
    // }
    //
    // robot.AutoAim.USB_RxCpltCallback(Buffer, Length);
}


/**
 *@brief TIM4任务回调函数
 *
 */
void Task100us_TIM4_Callback()
{

}

/**
 *@brief TIM5任务回调函数
 *
 */
void Task1ms_TIM7_Callback()
{
    // 在 10/100ms 状态机推进前，先刷新 XY 实测位置并处理待发送的停机命令。
    robot.TIM_1ms_Calculate_Callback();

    // 模块保持存活

    static int alive_mod100 = 0;
    alive_mod100++;
    if (alive_mod100 == 100)
    {
        alive_mod100 = 0;
        robot.TIM_100ms_Alive_PeriodElapsedCallback();

        // 调试变量读取实际正在使用的Holding_Jaw TOF实例，
        // 避免原代码读取一个从未Init的全局TOF050F对象。
        tof_distance =
            robot.Holding_Jaw.TOF050F.Get_Distance_mm();


        tof_online =
            robot.Holding_Jaw.TOF050F.Get_Online();


    }

    static int alive_mod1000 = 0;
    alive_mod1000++;
    if (alive_mod1000 == 1000)
    {
        alive_mod1000 = 0;
        robot.TIM_1000ms_Alive_PeriodElapsedCallback();

    }

    static int interaction_mod200 = 0;
    interaction_mod200++;
    if (interaction_mod200 == 200)
    {
        interaction_mod200 = 0;
        robot.TIM_200ms_Alive_PeriodElapsedCallback();
    }

    static int interaction_mod100 = 0;
    interaction_mod100++;
    if (interaction_mod100 == 100)
    {
        interaction_mod100 = 0;
        robot.TIM_100ms_Calculate_Callback();
    }

    static int interaction_mod10 = 0;
    interaction_mod10++;
    if (interaction_mod10 == 10)
    {
        interaction_mod10 = 0;
        robot.TIM_10ms_Calculate_PeriodElapsedCallback();
    }

    static int interaction_mod2 = 0;
    interaction_mod2++;
    if (interaction_mod2 == 2)
    {
        interaction_mod2 = 0;
        robot.TIM_2ms_Calculate_PeriodElapsedCallback();
    }

    // 设备层回调函数
    // Vofa_JustFloat_UART1_Callback();
    // Vofa_JustFloat.TIM_1ms_PeriodElapsedCallback();

    // 驱动层回调函数
    // TIM_1ms_CAN_PeriodElapsedCallback();

    // DM_Motor.TIM_Send_PeriodElapsedCallback();
    a = robot.Holding_Jaw.XY_Motion.Get_Power_On_Origin_Captured();
    b = robot.Holding_Jaw.XY_Motion.Get_Position_Valid();
    c = robot.Holding_Jaw.XY_Motion.Get_Fault();
    d = robot.Holding_Jaw.XY_Motion.Get_Stop_Confirmed();
}

/**
 *@brief 初始化任务
 *
 */
void Task_Init()
{
    init_finished = false;

    SYS_Timestamp.Init(
        &htim5
    );

    UART_Init(
        &huart2,
        TOF050F_UART2_Callback,
        32
    );

    UART_Init(
        &huart3,
        Servo_UART3_Callback,
        64
    );

    UART_Init(
        &huart4,
        Servo_UART4_Callback,
        64
    );


    robot.Holding_Jaw.TOF050F.Init(
        &huart2,
        0x01
    );


    // ========================================================
    // CAN启动
    // ========================================================

    CAN_Init(
        &hfdcan1,
        Device_CAN1_Callback
    );
    if (XY_Config::CAN_Bus() == &hfdcan2)
    {
        CAN_Init(&hfdcan2, Device_CAN2_Callback);
    }
    else if (XY_Config::CAN_Bus() == &hfdcan3)
    {
        CAN_Init(&hfdcan3, Device_CAN3_Callback);
    }


    HAL_TIM_Base_Start(
        &htim5
    );


    // ========================================================
    // 给DM4310等外设留启动时间
    // ========================================================

    HAL_Delay(
        2000
    );


    // ========================================================
    // 初始化整机对象
    //
    // Sorting_Storage.Init内部会先尝试一次：
    // 清除错误并重新使能
    // ========================================================

    robot.Init();

    robot.Init_Sorting_Communication(
        &huart8
    );


    robot.Holding_Jaw
        .TOF050F
        .Set_Auto_Request(
            true
        );


    Vofa_JustFloat_Init();

    Vofa_Tuner_Init();


    // ========================================================
    // 重要：
    // 对象全部Init完成后立即允许周期任务
    // ========================================================

    init_finished =
        true;


    // ========================================================
    // 最后启动TIM7
    // ========================================================

    HAL_TIM_Base_Start_IT(
        &htim7
    );


    HAL_Delay(1000);
    // robot.Holding_Jaw.XY_Motion.Move_To_mm(-170.0F,0.0F,15.0F,15.0F);
    // 不再上电自动Start：正式流程由PC1实体按键启动；
    // 保持STOPPED也允许Jetson执行独立XY标定命令。
}

/**
 *@brief 前台循环任务
 *
 */
void Task_Loop()
{
    // 等待 CAN 反馈刷新或机构空闲时，每 10ms 重试人工请求。
    // 一旦报错，必须等待新的操作员请求，不能自动重启。
    static uint32_t origin_attempt_tick = 0U;
    if (XY_Origin_Confirm_Request != 0U &&
        (uint32_t)(HAL_GetTick() - origin_attempt_tick) >= 10U)
    {
        origin_attempt_tick = HAL_GetTick();
        const uint32_t origin_irq_state = __get_PRIMASK();
        __disable_irq();
        const float origin_x_mm = XY_Origin_Confirm_X_mm;
        const float origin_y_mm = XY_Origin_Confirm_Y_mm;
        const HAL_StatusTypeDef status = robot.Confirm_XY_Origin_mm(origin_x_mm, origin_y_mm);
        XY_Origin_Confirm_Status = status;
        if (status != HAL_BUSY)
        {
            XY_Origin_Confirm_Request = 0U;
        }
        __set_PRIMASK(origin_irq_state);
    }

    // 自动碰撞回零必须由调试器/上位机显式请求，默认配置关闭。
    // 请求在状态机真正接受或明确失败后清零；HAL_BUSY 保留请求等待机构空闲。
    static uint32_t homing_attempt_tick = 0U;
    if (XY_Collision_Homing_Request != 0U &&
        (uint32_t)(HAL_GetTick() - homing_attempt_tick) >= 10U)
    {
        homing_attempt_tick = HAL_GetTick();
        const HAL_StatusTypeDef status = robot.Start_XY_Collision_Homing();
        XY_Collision_Homing_Status = status;
        if (status != HAL_BUSY) XY_Collision_Homing_Request = 0U;
    }

    // ========================================================
    // 实体按键 PC1（高电平按下，外部/内部下拉）
    //
    // 修正版语义：
    // 1) STOPPED：按下稳定 30ms 后立即锁存 START，
    //    不再等到“松手”才启动，避免短按被丢失。
    // 2) RUNNING / 独立标定 / 储物测试 / START回位：短按完全忽略。
    // 3) RUNNING / 标定：只有“在这些状态中新按下”的这一按，
    //    连续保持 >= 1.5s 才触发紧急 STOP。
    // 4) 用于启动的那一次按键，即使启动后继续按住，
    //    也绝不会在 1.5s 后反过来触发 STOP。
    // 5) 按下/松开均采用 30ms 稳定去抖。
    // ========================================================

    static GPIO_PinState last_raw_state = GPIO_PIN_RESET;
    static GPIO_PinState stable_state = GPIO_PIN_RESET;

    // 上电时如果按键已经是高电平，不能把“上电持键/输入误高”误认为一次
    // START。必须先观察到稳定松开，再接受下一次真正的按下沿。
    static bool power_on_release_required = true;

    // 最近一次原始电平变化的时刻。
    static uint32_t raw_change_tick = 0U;

    // 用于 RUNNING 或独立XY/Yaw标定的长按STOP计时。
    static uint32_t press_tick = 0U;

    // 这一按是否开始于RUNNING或独立标定期间。
    // 只有它为 true，才允许长按触发 STOP。
    static bool press_started_with_active_motion = false;

    // 同一次持续按住只允许发一次 STOP。
    static bool long_stop_sent = false;

    static constexpr uint32_t KEY_PRESS_DEBOUNCE_MS = 30U;
    static constexpr uint32_t KEY_RELEASE_DEBOUNCE_MS = 30U;
    static constexpr uint32_t KEY_EMERGENCY_STOP_HOLD_MS = 1500U;

    const uint32_t now = HAL_GetTick();

    const GPIO_PinState raw_state = HAL_GPIO_ReadPin(
        SORT_KEY_GPIO_Port,
        SORT_KEY_Pin
    );

    // --------------------------------------------------------
    // 原始电平一变化，只更新时间戳；
    // 必须保持同一电平达到去抖时间后，才改变 stable_state。
    // --------------------------------------------------------
    if (raw_state != last_raw_state)
    {
        last_raw_state = raw_state;
        raw_change_tick = now;
    }

    const uint32_t required_debounce_ms =
        (raw_state == GPIO_PIN_SET)
            ? KEY_PRESS_DEBOUNCE_MS
            : KEY_RELEASE_DEBOUNCE_MS;

    if (
        stable_state != raw_state
        &&
        (uint32_t)(now - raw_change_tick) >= required_debounce_ms
    )
    {
        stable_state = raw_state;

        // ====================================================
        // 稳定按下沿
        // ====================================================
        if (stable_state == GPIO_PIN_SET)
        {
            const bool was_running = robot.Get_Sorting_Running();
            const bool was_active = robot.Get_Mechanism_Action_Active();

            press_tick = now;
            press_started_with_active_motion = was_active;
            long_stop_sent = false;

            if (!power_on_release_required && !was_running && !was_active)
            {
                // 关键修正：
                // START 在“稳定按下沿”立即锁存，而不是等稳定松开。
                // Request_Sorting_Start() 内部仍会检查 STOPPED 和同步状态，
                // 一旦请求被接受，机构暂时未 Ready 时会继续锁存等待。
                robot.Request_Sorting_Start();
            }
            else if (power_on_release_required)
            {
                // 上电时按键已经是高电平，先忽略这一按，避免误启动回待机。
                press_started_with_active_motion = false;
            }
            // RUNNING或标定期间：短按不做任何事；
            // 只有后面的 1.5s 长按逻辑才可能 STOP。
        }
        // ====================================================
        // 稳定松开沿
        // ====================================================
        else
        {
            // 松开只负责结束本次按键周期，绝不在这里产生 START。
            power_on_release_required = false;
            press_started_with_active_motion = false;
            long_stop_sent = false;
        }
    }

    // ========================================================
    // RUNNING / 独立XY、Yaw标定长按STOP
    //
    // 必须同时满足：
    // - 当前仍稳定按下；
    // - 这一按是在RUNNING或独立标定时开始的；
    // - 持续 >= 1.5s；
    // - 本次按住尚未发送过 STOP。
    //
    // 因此：启动用的那一次按键即使一直不松手，也不会触发 STOP。
    // ========================================================
    if (
        stable_state == GPIO_PIN_SET
        &&
        press_started_with_active_motion
        &&
        !long_stop_sent
        &&
        (uint32_t)(now - press_tick) >= KEY_EMERGENCY_STOP_HOLD_MS
    )
    {
        robot.Stop_Sorting_Work();
        long_stop_sent = true;
    }
}



/**
 * @brief 定时器中断回调函数
 *
 * @param htim
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (!init_finished)
    {
        return;
    }

    if (htim->Instance == TIM5)
    {
        Task3600s_Callback();
    }
    else if (htim->Instance == TIM7)
    {
        Task1ms_TIM7_Callback();
    }

}
