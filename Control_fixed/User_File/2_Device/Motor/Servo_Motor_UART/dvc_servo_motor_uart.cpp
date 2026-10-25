/**
 * @file dvc_servo_motor_uart.cpp
 * @brief 众灵 ZP 系列单总线舵机 UART 驱动
 * @version 1.0
 * @date 2026-08-07
 */

/* Includes ------------------------------------------------------------------*/

#include "2_Device/Motor/Servo_Motor_UART/dvc_servo_motor_uart.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

/* Private macros ------------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private variables ---------------------------------------------------------*/

/* Private function declarations ---------------------------------------------*/

/* function prototypes -------------------------------------------------------*/

/**
 * @brief 初始化众灵总线舵机
 */
void Class_Servo_Motor::Init(UART_HandleTypeDef *huart,
                             uint8_t id,
                             Enum_Servo_Motor_Work_Mode __Work_Mode)
{
    UART_Handler = huart;
    Servo_Motor_ID = (id <= 254U) ? id : 254U;
    Pending_ID = 0xFF;

    Rx_Data = Struct_Servo_Motor_UART_Rx_Data{};
    Rx_Data.Work_Mode = __Work_Mode;

    Flag = 0;
    Pre_Flag = 0;
    Servo_Motor_Status = Servo_Motor_Status_DISABLE;

    Rx_Frame_Length = 0;
    Rx_In_Frame = 0;
    memset(Rx_Frame_Buffer, 0, sizeof(Rx_Frame_Buffer));
    memset(Last_Response, 0, sizeof(Last_Response));
    memset(Tx_Buffer, 0, sizeof(Tx_Buffer));

    Update_Angle_Range();
    Target_Angle = 0.0f;
    Target_PWM = SERVO_MOTOR_UART_PWM_MID;
}

/**
 * @brief 使用当前 Target_Angle 发送位置控制
 */
uint8_t Class_Servo_Motor::Control_Angle(const uint16_t &time_ms)
{
    if (Rx_Data.Work_Mode < Servo_Motor_Work_Mode_270_CW ||
        Rx_Data.Work_Mode > Servo_Motor_Work_Mode_180_CCW)
    {
        return HAL_ERROR;
    }

    Target_PWM = Angle_To_PWM(Target_Angle);
    return Transmit_Control(Target_PWM, Constrain_U16(time_ms, 0U, SERVO_MOTOR_UART_TIME_MAX));
}

/**
 * @brief 设置目标角度并立即发送位置控制
 */
uint8_t Class_Servo_Motor::Control_Angle(const float &__Target_Angle, const uint16_t &time_ms)
{
    Set_Target_Angle(__Target_Angle);
    return Control_Angle(time_ms);
}

/**
 * @brief 直接按协议 P/T 字段发送控制
 */
uint8_t Class_Servo_Motor::Control_PWM(const uint16_t &pwm_or_speed, const uint16_t &time_or_count)
{
    Target_PWM = Constrain_U16(pwm_or_speed, SERVO_MOTOR_UART_PWM_MIN, SERVO_MOTOR_UART_PWM_MAX);

    if (Rx_Data.Work_Mode >= Servo_Motor_Work_Mode_270_CW &&
        Rx_Data.Work_Mode <= Servo_Motor_Work_Mode_180_CCW)
    {
        Target_Angle = PWM_To_Angle(Target_PWM);
    }

    return Transmit_Control(Target_PWM,
                            Constrain_U16(time_or_count, 0U, SERVO_MOTOR_UART_TIME_MAX));
}

/**
 * @brief 马达模式控制
 */
uint8_t Class_Servo_Motor::Control_Motor(const int16_t &speed_offset, const uint16_t &time_or_count)
{
    if (Rx_Data.Work_Mode < Servo_Motor_Work_Mode_MOTOR_ROUND_CW ||
        Rx_Data.Work_Mode > Servo_Motor_Work_Mode_MOTOR_TIME_CCW)
    {
        return HAL_ERROR;
    }

    int16_t constrained_speed = Constrain_I16(speed_offset, -1000, 1000);
    uint16_t p_value = static_cast<uint16_t>(SERVO_MOTOR_UART_PWM_MID + constrained_speed);
    return Control_PWM(p_value, time_or_count);
}

uint8_t Class_Servo_Motor::Read_Version()
{
    return Transmit_Command("PVER");
}

uint8_t Class_Servo_Motor::Read_ID()
{
    return Transmit_Command("PID");
}

uint8_t Class_Servo_Motor::Set_ID(const uint8_t &new_id)
{
    if (new_id > 254U)
    {
        return HAL_ERROR;
    }

    if (UART_Handler == nullptr)
    {
        return HAL_ERROR;
    }

    // DMA 正在发送时不能覆盖 Tx_Buffer
    if (UART_Handler->gState != HAL_UART_STATE_READY)
    {
        return HAL_BUSY;
    }

    int length = snprintf(reinterpret_cast<char *>(Tx_Buffer),
                          sizeof(Tx_Buffer),
                          "#%03uPID%03u!",
                          static_cast<unsigned int>(Servo_Motor_ID),
                          static_cast<unsigned int>(new_id));

    if (length <= 0 || length >= static_cast<int>(sizeof(Tx_Buffer)))
    {
        return HAL_ERROR;
    }

    Pending_ID = new_id;
    return UART_Transmit_Data(UART_Handler, Tx_Buffer, static_cast<uint16_t>(length));
}

uint8_t Class_Servo_Motor::Release_Torque()
{
    return Transmit_Command("PULK");
}

uint8_t Class_Servo_Motor::Recover_Torque()
{
    return Transmit_Command("PULR");
}

uint8_t Class_Servo_Motor::Read_Work_Mode()
{
    return Transmit_Command("PMOD");
}

uint8_t Class_Servo_Motor::Set_Work_Mode(const Enum_Servo_Motor_Work_Mode &mode)
{
    if (mode < Servo_Motor_Work_Mode_270_CW || mode > Servo_Motor_Work_Mode_MOTOR_TIME_CCW)
    {
        return HAL_ERROR;
    }

    char body[8] = {0};
    snprintf(body, sizeof(body), "PMOD%u", static_cast<unsigned int>(mode));
    uint8_t status = Transmit_Command(body);

    if (status == HAL_OK)
    {
        // 先更新本地模式，收到 PMOD 回读后还会再次校准
        Rx_Data.Work_Mode = mode;
        Update_Angle_Range();
        Set_Target_PWM(Target_PWM);
    }

    return status;
}

uint8_t Class_Servo_Motor::Read_Angle()
{
    return Transmit_Command("PRAD");
}

uint8_t Class_Servo_Motor::Pause()
{
    return Transmit_Command("PDPT");
}

uint8_t Class_Servo_Motor::Continue()
{
    return Transmit_Command("PDCT");
}

uint8_t Class_Servo_Motor::Stop()
{
    return Transmit_Command("PDST");
}

uint8_t Class_Servo_Motor::Set_Baud_Rate(const Enum_Servo_Motor_Baud_Rate &baud_rate)
{
    if (baud_rate < Servo_Motor_Baud_Rate_9600 || baud_rate > Servo_Motor_Baud_Rate_1000000)
    {
        return HAL_ERROR;
    }

    char body[8] = {0};
    snprintf(body, sizeof(body), "PBD%u", static_cast<unsigned int>(baud_rate));
    return Transmit_Command(body);
}

uint8_t Class_Servo_Motor::Correct_Middle_Value()
{
    return Transmit_Command("PSCK");
}

uint8_t Class_Servo_Motor::Set_Start_Value()
{
    return Transmit_Command("PCSD");
}

uint8_t Class_Servo_Motor::Start_With_Release_Torque()
{
    return Transmit_Command("PCSM");
}

uint8_t Class_Servo_Motor::Start_With_Recover_Torque()
{
    return Transmit_Command("PCSR");
}

uint8_t Class_Servo_Motor::Set_Min_Value()
{
    return Transmit_Command("PSMI");
}

uint8_t Class_Servo_Motor::Set_Max_Value()
{
    return Transmit_Command("PSMX");
}

uint8_t Class_Servo_Motor::Factory_Reset_Keep_ID()
{
    return Transmit_Command("PCLE0");
}

uint8_t Class_Servo_Motor::Factory_Reset_All()
{
    return Transmit_Command("PCLE");
}

uint8_t Class_Servo_Motor::Read_Temperature_And_Voltage()
{
    return Transmit_Command("PRTE");
}

/**
 * @brief UART 接收回调
 */
void Class_Servo_Motor::UART_RxCpltCallback(uint8_t *Buffer, uint16_t Length)
{
    if (Buffer == nullptr || Length == 0U)
    {
        return;
    }

    for (uint16_t i = 0; i < Length; i++)
    {
        Parse_Byte(Buffer[i]);
    }
}

/**
 * @brief 100ms 在线检测
 */
void Class_Servo_Motor::TIM_100ms_Alive_PeriodElapsedCallback()
{
    if (Flag == Pre_Flag)
    {
        Servo_Motor_Status = Servo_Motor_Status_DISABLE;
    }
    else
    {
        Servo_Motor_Status = Servo_Motor_Status_ENABLE;
    }

    Pre_Flag = Flag;
}

/**
 * @brief 发送无参数/短参数指令，例如 PMOD、PRAD、PDST
 */
uint8_t Class_Servo_Motor::Transmit_Command(const char *body)
{
    if (UART_Handler == nullptr || body == nullptr)
    {
        return HAL_ERROR;
    }

    // HAL_UART_Transmit_DMA 尚未结束时不能覆盖成员 Tx_Buffer
    if (UART_Handler->gState != HAL_UART_STATE_READY)
    {
        return HAL_BUSY;
    }

    int length = snprintf(reinterpret_cast<char *>(Tx_Buffer),
                          sizeof(Tx_Buffer),
                          "#%03u%s!",
                          static_cast<unsigned int>(Servo_Motor_ID),
                          body);

    if (length <= 0 || length >= static_cast<int>(sizeof(Tx_Buffer)))
    {
        return HAL_ERROR;
    }

    return UART_Transmit_Data(UART_Handler, Tx_Buffer, static_cast<uint16_t>(length));
}

/**
 * @brief 发送控制命令 #IDPxxxxTxxxx!
 */
uint8_t Class_Servo_Motor::Transmit_Control(const uint16_t &pwm_or_speed, const uint16_t &time_or_count)
{
    if (UART_Handler == nullptr)
    {
        return HAL_ERROR;
    }

    if (UART_Handler->gState != HAL_UART_STATE_READY)
    {
        return HAL_BUSY;
    }

    uint16_t p_value = Constrain_U16(pwm_or_speed,
                                     SERVO_MOTOR_UART_PWM_MIN,
                                     SERVO_MOTOR_UART_PWM_MAX);
    uint16_t t_value = Constrain_U16(time_or_count, 0U, SERVO_MOTOR_UART_TIME_MAX);

    int length = snprintf(reinterpret_cast<char *>(Tx_Buffer),
                          sizeof(Tx_Buffer),
                          "#%03uP%04uT%04u!",
                          static_cast<unsigned int>(Servo_Motor_ID),
                          static_cast<unsigned int>(p_value),
                          static_cast<unsigned int>(t_value));

    if (length <= 0 || length >= static_cast<int>(sizeof(Tx_Buffer)))
    {
        return HAL_ERROR;
    }

    return UART_Transmit_Data(UART_Handler, Tx_Buffer, static_cast<uint16_t>(length));
}

/**
 * @brief 串口流式拼帧，处理 ReceiveToIdle DMA 的拆包与粘包
 */
void Class_Servo_Motor::Parse_Byte(const uint8_t &data)
{
    if (data == '#')
    {
        Rx_In_Frame = 1;
        Rx_Frame_Length = 0;
        memset(Rx_Frame_Buffer, 0, sizeof(Rx_Frame_Buffer));
        Rx_Frame_Buffer[Rx_Frame_Length++] = '#';
        return;
    }

    if (!Rx_In_Frame)
    {
        return;
    }

    if (data < 0x20U || data > 0x7EU)
    {
        return;
    }

    if (Rx_Frame_Length >= RX_FRAME_BUFFER_SIZE - 1U)
    {
        Rx_In_Frame = 0;
        Rx_Frame_Length = 0;
        memset(Rx_Frame_Buffer, 0, sizeof(Rx_Frame_Buffer));
        return;
    }

    Rx_Frame_Buffer[Rx_Frame_Length++] = static_cast<char>(data);

    if (data == '!')
    {
        Rx_Frame_Buffer[Rx_Frame_Length] = '\0';
        Data_Process(Rx_Frame_Buffer);

        Rx_In_Frame = 0;
        Rx_Frame_Length = 0;
    }
}

/**
 * @brief 处理一帧舵机回包
 */
void Class_Servo_Motor::Data_Process(const char *frame)
{
    if (frame == nullptr || frame[0] != '#')
    {
        return;
    }

    size_t length = strlen(frame);
    if (length < 4U || frame[length - 1U] != '!')
    {
        return;
    }

    strncpy(Last_Response, frame, sizeof(Last_Response) - 1U);
    Last_Response[sizeof(Last_Response) - 1U] = '\0';

    // 大多数配置指令返回 #OK!，该格式不带 ID
    if (strcmp(frame, "#OK!") == 0)
    {
        Flag++;
        Servo_Motor_Status = Servo_Motor_Status_ENABLE;
        return;
    }

    // 后续带 ID 的回包至少形如 #000P!
    if (length < 6U ||
        !isdigit(static_cast<unsigned char>(frame[1])) ||
        !isdigit(static_cast<unsigned char>(frame[2])) ||
        !isdigit(static_cast<unsigned char>(frame[3])))
    {
        return;
    }

    uint16_t rx_id = static_cast<uint16_t>((frame[1] - '0') * 100 +
                                           (frame[2] - '0') * 10 +
                                           (frame[3] - '0'));

    // Set_ID 成功返回新 ID，例如 #001P!
    if (Pending_ID <= 254U && rx_id == Pending_ID && frame[4] == 'P' && frame[5] == '!')
    {
        Servo_Motor_ID = Pending_ID;
        Pending_ID = 0xFF;
        Flag++;
        Servo_Motor_Status = Servo_Motor_Status_ENABLE;
        return;
    }

    // 忽略其他舵机的回包；广播 ID 只用于发送，不用于绑定对象
    if (rx_id != Servo_Motor_ID)
    {
        return;
    }

    Flag++;
    Servo_Motor_Status = Servo_Motor_Status_ENABLE;

    // #000P!：ID 检测成功
    if (frame[4] == 'P' && frame[5] == '!')
    {
        return;
    }

    // #000P1500!：位置回读
    if (frame[4] == 'P' && length >= 10U &&
        isdigit(static_cast<unsigned char>(frame[5])) &&
        isdigit(static_cast<unsigned char>(frame[6])) &&
        isdigit(static_cast<unsigned char>(frame[7])) &&
        isdigit(static_cast<unsigned char>(frame[8])) &&
        frame[9] == '!')
    {
        uint16_t pwm = static_cast<uint16_t>((frame[5] - '0') * 1000 +
                                             (frame[6] - '0') * 100 +
                                             (frame[7] - '0') * 10 +
                                             (frame[8] - '0'));
        Rx_Data.Now_PWM = Constrain_U16(pwm,
                                        SERVO_MOTOR_UART_PWM_MIN,
                                        SERVO_MOTOR_UART_PWM_MAX);
        Rx_Data.Now_Angle = PWM_To_Angle(Rx_Data.Now_PWM);
        return;
    }

    // #000PMOD1!：工作模式回读/设置回包
    if (frame[4] == 'P' && strncmp(&frame[5], "MOD", 3U) == 0 &&
        length >= 10U && frame[8] >= '1' && frame[8] <= '8')
    {
        Rx_Data.Work_Mode = static_cast<Enum_Servo_Motor_Work_Mode>(frame[8] - '0');
        Update_Angle_Range();
        Rx_Data.Now_Angle = PWM_To_Angle(Rx_Data.Now_PWM);
        Target_Angle = PWM_To_Angle(Target_PWM);
        return;
    }

    // #000PV0.97!：版本号
    if (frame[4] == 'P' && frame[5] == 'V')
    {
        size_t copy_length = length - 6U; // 去掉 #dddP 和 !
        if (copy_length >= sizeof(Rx_Data.Version))
        {
            copy_length = sizeof(Rx_Data.Version) - 1U;
        }

        memcpy(Rx_Data.Version, &frame[5], copy_length);
        Rx_Data.Version[copy_length] = '\0';
        return;
    }

    // #000T1927-08.1!：温度/电压回读。
    // 手册只给出示例，没有说明 1927 的温度缩放关系，因此温度保留原始整数值。
    if (frame[4] == 'T')
    {
        const char *temperature_begin = &frame[5];
        char *temperature_end = nullptr;
        long temperature_raw = strtol(temperature_begin, &temperature_end, 10);

        if (temperature_end != temperature_begin)
        {
            Rx_Data.Temperature_Raw = static_cast<int32_t>(temperature_raw);
        }

        const char *separator = strchr(temperature_begin, '-');
        if (separator != nullptr && separator[1] != '\0')
        {
            char *voltage_end = nullptr;
            float voltage = strtof(separator + 1, &voltage_end);
            if (voltage_end != separator + 1)
            {
                Rx_Data.Now_Voltage = voltage;
            }
        }
        return;
    }
}

/**
 * @brief 根据 PMOD 更新角度范围
 */
void Class_Servo_Motor::Update_Angle_Range()
{
    switch (Rx_Data.Work_Mode)
    {
    case Servo_Motor_Work_Mode_270_CW:
    case Servo_Motor_Work_Mode_270_CCW:
        Min_Angle = -135.0f * PI / 180.0f;
        Max_Angle = 135.0f * PI / 180.0f;
        break;

    case Servo_Motor_Work_Mode_180_CW:
    case Servo_Motor_Work_Mode_180_CCW:
        Min_Angle = -90.0f * PI / 180.0f;
        Max_Angle = 90.0f * PI / 180.0f;
        break;

    default:
        // 马达模式没有位置控制含义，保留 PWM 原始控制接口
        Min_Angle = 0.0f;
        Max_Angle = 0.0f;
        break;
    }
}

/**
 * @brief rad -> P 值，P1500 为 0 rad
 */
uint16_t Class_Servo_Motor::Angle_To_PWM(const float &angle) const
{
    if (Rx_Data.Work_Mode > Servo_Motor_Work_Mode_180_CCW)
    {
        return SERVO_MOTOR_UART_PWM_MID;
    }

    float constrained_angle = Constrain_Float(angle, Min_Angle, Max_Angle);
    float half_range = Max_Angle;

    if (half_range <= 0.0f)
    {
        return SERVO_MOTOR_UART_PWM_MID;
    }

    float normalized = constrained_angle / half_range;

    // PMOD2/4 为逆时针模式，P 值方向相反
    if (Rx_Data.Work_Mode == Servo_Motor_Work_Mode_270_CCW ||
        Rx_Data.Work_Mode == Servo_Motor_Work_Mode_180_CCW)
    {
        normalized = -normalized;
    }

    float pwm = static_cast<float>(SERVO_MOTOR_UART_PWM_MID) + normalized * 1000.0f;
    pwm = Constrain_Float(pwm,
                          static_cast<float>(SERVO_MOTOR_UART_PWM_MIN),
                          static_cast<float>(SERVO_MOTOR_UART_PWM_MAX));

    return static_cast<uint16_t>(pwm + 0.5f);
}

/**
 * @brief P 值 -> rad，P1500 为 0 rad
 */
float Class_Servo_Motor::PWM_To_Angle(const uint16_t &pwm) const
{
    if (Rx_Data.Work_Mode > Servo_Motor_Work_Mode_180_CCW)
    {
        return 0.0f;
    }

    uint16_t constrained_pwm = Constrain_U16(pwm,
                                              SERVO_MOTOR_UART_PWM_MIN,
                                              SERVO_MOTOR_UART_PWM_MAX);
    float normalized = (static_cast<float>(constrained_pwm) -
                        static_cast<float>(SERVO_MOTOR_UART_PWM_MID)) / 1000.0f;

    if (Rx_Data.Work_Mode == Servo_Motor_Work_Mode_270_CCW ||
        Rx_Data.Work_Mode == Servo_Motor_Work_Mode_180_CCW)
    {
        normalized = -normalized;
    }

    return normalized * Max_Angle;
}

uint16_t Class_Servo_Motor::Constrain_U16(const uint16_t &value,
                                          const uint16_t &min_value,
                                          const uint16_t &max_value)
{
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

int16_t Class_Servo_Motor::Constrain_I16(const int16_t &value,
                                         const int16_t &min_value,
                                         const int16_t &max_value)
{
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

float Class_Servo_Motor::Constrain_Float(const float &value,
                                         const float &min_value,
                                         const float &max_value)
{
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}
