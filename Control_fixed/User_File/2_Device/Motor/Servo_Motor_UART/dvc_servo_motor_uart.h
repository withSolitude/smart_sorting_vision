/**
 * @file dvc_servo_motor_uart.h
 * @brief 众灵 ZP 系列单总线舵机 UART 驱动
 * @version 1.0
 * @date 2026-08-07
 *
 * @note
 * 1. 协议来源：众灵舵机使用手册（250508），默认波特率 115200。
 * 2. 舵机协议为 ASCII 字符串，无校验，例如 #000P1500T1000!。
 * 3. 本驱动只负责设备层协议封装；UART DMA 接收仍由 drv_uart 管理，
 *    在对应 UART 回调中调用 UART_RxCpltCallback() 即可。
 * 4. 角度接口统一使用 rad，并以 P1500 为机械中位 0 rad。
 */

#ifndef DVC_SERVO_MOTOR_UART_H
#define DVC_SERVO_MOTOR_UART_H

/* Includes ------------------------------------------------------------------*/

#include <cstdint>
#include "1_Middleware/1_Driver/UART/drv_uart.h"
#include "arm_math.h"
/* Exported macros -----------------------------------------------------------*/

#define SERVO_MOTOR_UART_PWM_MIN              500U
#define SERVO_MOTOR_UART_PWM_MID              1500U
#define SERVO_MOTOR_UART_PWM_MAX              2500U
#define SERVO_MOTOR_UART_TIME_MAX             9999U
#define SERVO_MOTOR_UART_BROADCAST_ID          255U
#define SERVO_MOTOR_UART_DEFAULT_BAUD_RATE     115200U

/* Exported types ------------------------------------------------------------*/

/**
 * @brief 舵机在线状态
 */
enum Enum_Servo_Motor_Status : uint8_t
{
    Servo_Motor_Status_DISABLE = 0,
    Servo_Motor_Status_ENABLE,
};

/**
 * @brief 舵机工作模式，对应手册 PMOD1 ~ PMOD8
 */
enum Enum_Servo_Motor_Work_Mode : uint8_t
{
    Servo_Motor_Work_Mode_270_CW = 1,
    Servo_Motor_Work_Mode_270_CCW,
    Servo_Motor_Work_Mode_180_CW,
    Servo_Motor_Work_Mode_180_CCW,
    Servo_Motor_Work_Mode_MOTOR_ROUND_CW,
    Servo_Motor_Work_Mode_MOTOR_ROUND_CCW,
    Servo_Motor_Work_Mode_MOTOR_TIME_CW,
    Servo_Motor_Work_Mode_MOTOR_TIME_CCW,
};

// 兼容原文件中的拼写
using Enum_Servo_Motor_Woke_Mode = Enum_Servo_Motor_Work_Mode;

/**
 * @brief 波特率设置参数，对应手册 PBD0 ~ PBD7
 */
enum Enum_Servo_Motor_Baud_Rate : uint8_t
{
    Servo_Motor_Baud_Rate_9600 = 0,
    Servo_Motor_Baud_Rate_19200,
    Servo_Motor_Baud_Rate_38400,
    Servo_Motor_Baud_Rate_57600,
    Servo_Motor_Baud_Rate_115200,
    Servo_Motor_Baud_Rate_128000,
    Servo_Motor_Baud_Rate_256000,
    Servo_Motor_Baud_Rate_1000000,
};

/**
 * @brief 舵机回读数据
 */
struct Struct_Servo_Motor_UART_Rx_Data
{
    // PRAD 返回的原始位置值，500 ~ 2500
    uint16_t Now_PWM = SERVO_MOTOR_UART_PWM_MID;

    // 根据工作模式换算后的当前角度，rad，P1500 对应 0 rad
    float Now_Angle = 0.0f;

    // PRTE 示例中温度字段为 4 位数，但手册未给出缩放关系，因此保留原始值
    int32_t Temperature_Raw = 0;

    // PRTE 返回电压，V
    float Now_Voltage = 0.0f;

    // 当前工作模式
    Enum_Servo_Motor_Work_Mode Work_Mode = Servo_Motor_Work_Mode_270_CW;

    // 版本号字符串，例如 "V0.97"
    char Version[16] = {0};
};

/**
 * @brief 众灵 ZP 系列单总线舵机
 */
class Class_Servo_Motor
{
public:
    /**
     * @brief 初始化舵机对象
     *
     * @param huart 绑定 UART
     * @param id 舵机 ID，0 ~ 254
     * @param __Work_Mode 本地记录的工作模式，默认 PMOD1
     */
    void Init(UART_HandleTypeDef *huart,
              uint8_t id,
              Enum_Servo_Motor_Work_Mode __Work_Mode = Servo_Motor_Work_Mode_270_CW);

    inline Enum_Servo_Motor_Status Get_Status() const;
    inline uint8_t Get_ID() const;
    inline Enum_Servo_Motor_Work_Mode Get_Work_Mode() const;
    inline uint16_t Get_Now_PWM() const;
    inline float Get_Now_Angle() const;
    inline int32_t Get_Temperature_Raw() const;
    inline float Get_Now_Voltage() const;
    inline const char *Get_Version() const;
    inline const char *Get_Last_Response() const;
    inline float Get_Target_Angle() const;
    inline uint16_t Get_Target_PWM() const;

    inline void Set_Target_Angle(const float &__Target_Angle);
    inline void Set_Target_PWM(const uint16_t &__Target_PWM);

    /**
     * @brief 按目标角度控制舵机，仅适用于 PMOD1 ~ PMOD4
     * @param time_ms 运动时间，0 ~ 9999 ms
     */
    uint8_t Control_Angle(const uint16_t &time_ms = 0);
    uint8_t Control_Angle(const float &__Target_Angle, const uint16_t &time_ms);

    /**
     * @brief 直接按 P 值发送控制命令
     * @param pwm_or_speed P 字段，500 ~ 2500
     * @param time_or_count T 字段；舵机模式为 ms，马达模式按手册解释为圈数/时间
     */
    uint8_t Control_PWM(const uint16_t &pwm_or_speed, const uint16_t &time_or_count = 0);

    /**
     * @brief 马达模式控制，speed_offset 范围 -1000 ~ 1000，对应 P=1500+offset
     */
    uint8_t Control_Motor(const int16_t &speed_offset, const uint16_t &time_or_count = 0);

    // 配置/读取指令
    uint8_t Read_Version();
    uint8_t Read_ID();
    uint8_t Set_ID(const uint8_t &new_id);
    uint8_t Release_Torque();
    uint8_t Recover_Torque();
    uint8_t Read_Work_Mode();
    uint8_t Set_Work_Mode(const Enum_Servo_Motor_Work_Mode &mode);
    uint8_t Read_Angle();
    uint8_t Pause();
    uint8_t Continue();
    uint8_t Stop();
    uint8_t Set_Baud_Rate(const Enum_Servo_Motor_Baud_Rate &baud_rate);
    uint8_t Correct_Middle_Value();
    uint8_t Set_Start_Value();
    uint8_t Start_With_Release_Torque();
    uint8_t Start_With_Recover_Torque();
    uint8_t Set_Min_Value();
    uint8_t Set_Max_Value();
    uint8_t Factory_Reset_Keep_ID();
    uint8_t Factory_Reset_All();
    uint8_t Read_Temperature_And_Voltage();

    /**
     * @brief UART 接收回调，将 drv_uart 的 Buffer/Length 转发到这里
     */
    void UART_RxCpltCallback(uint8_t *Buffer, uint16_t Length);

    /**
     * @brief 100ms 在线检测；只有周期性读取反馈时该状态才有意义
     */
    void TIM_100ms_Alive_PeriodElapsedCallback();

protected:
    // 初始化相关变量

    UART_HandleTypeDef *UART_Handler = nullptr;
    uint8_t Servo_Motor_ID = 0;
    uint8_t Pending_ID = 0xFF;

    // 常量

    // static constexpr float PI = 3.14159265358979323846f;
    static constexpr uint16_t TX_BUFFER_SIZE = 64;
    static constexpr uint16_t RX_FRAME_BUFFER_SIZE = 64;

    // 内部变量

    uint32_t Flag = 0;
    uint32_t Pre_Flag = 0;

    Enum_Servo_Motor_Status Servo_Motor_Status = Servo_Motor_Status_DISABLE;
    Struct_Servo_Motor_UART_Rx_Data Rx_Data;

    float Min_Angle = -135.0f * PI / 180.0f;
    float Max_Angle = 135.0f * PI / 180.0f;
    float Target_Angle = 0.0f;
    uint16_t Target_PWM = SERVO_MOTOR_UART_PWM_MID;

    // DMA 发送期间 Tx_Buffer 必须保持有效，不能使用栈上临时数组
    uint8_t Tx_Buffer[TX_BUFFER_SIZE] = {0};

    // UART ReceiveToIdle 可能拆包/粘包，使用流式拼帧
    char Rx_Frame_Buffer[RX_FRAME_BUFFER_SIZE] = {0};
    uint16_t Rx_Frame_Length = 0;
    uint8_t Rx_In_Frame = 0;

    char Last_Response[RX_FRAME_BUFFER_SIZE] = {0};

    // 内部函数

    uint8_t Transmit_Command(const char *body);
    uint8_t Transmit_Control(const uint16_t &pwm_or_speed, const uint16_t &time_or_count);
    void Parse_Byte(const uint8_t &data);
    void Data_Process(const char *frame);
    void Update_Angle_Range();

    uint16_t Angle_To_PWM(const float &angle) const;
    float PWM_To_Angle(const uint16_t &pwm) const;

    static uint16_t Constrain_U16(const uint16_t &value, const uint16_t &min_value, const uint16_t &max_value);
    static int16_t Constrain_I16(const int16_t &value, const int16_t &min_value, const int16_t &max_value);
    static float Constrain_Float(const float &value, const float &min_value, const float &max_value);
};

/* Inline functions ----------------------------------------------------------*/

inline Enum_Servo_Motor_Status Class_Servo_Motor::Get_Status() const
{
    return (Servo_Motor_Status);
}

inline uint8_t Class_Servo_Motor::Get_ID() const
{
    return (Servo_Motor_ID);
}

inline Enum_Servo_Motor_Work_Mode Class_Servo_Motor::Get_Work_Mode() const
{
    return (Rx_Data.Work_Mode);
}

inline uint16_t Class_Servo_Motor::Get_Now_PWM() const
{
    return (Rx_Data.Now_PWM);
}

inline float Class_Servo_Motor::Get_Now_Angle() const
{
    return (Rx_Data.Now_Angle);
}

inline int32_t Class_Servo_Motor::Get_Temperature_Raw() const
{
    return (Rx_Data.Temperature_Raw);
}

inline float Class_Servo_Motor::Get_Now_Voltage() const
{
    return (Rx_Data.Now_Voltage);
}

inline const char *Class_Servo_Motor::Get_Version() const
{
    return (Rx_Data.Version);
}

inline const char *Class_Servo_Motor::Get_Last_Response() const
{
    return (Last_Response);
}

inline float Class_Servo_Motor::Get_Target_Angle() const
{
    return (Target_Angle);
}

inline uint16_t Class_Servo_Motor::Get_Target_PWM() const
{
    return (Target_PWM);
}

inline void Class_Servo_Motor::Set_Target_Angle(const float &__Target_Angle)
{
    Target_Angle = Constrain_Float(__Target_Angle, Min_Angle, Max_Angle);
    Target_PWM = Angle_To_PWM(Target_Angle);
}

inline void Class_Servo_Motor::Set_Target_PWM(const uint16_t &__Target_PWM)
{
    Target_PWM = Constrain_U16(__Target_PWM, SERVO_MOTOR_UART_PWM_MIN, SERVO_MOTOR_UART_PWM_MAX);
    Target_Angle = PWM_To_Angle(Target_PWM);
}

#endif // DVC_SERVO_MOTOR_UART_H
