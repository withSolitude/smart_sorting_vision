#include "2_Device/PC_Communication/dvc_sorting_vision_uart_v2.h"

#include <cmath>
#include <climits>
#include <cstring>


Class_Sorting_Vision_UART *
Class_Sorting_Vision_UART::Instance =
    nullptr;


// ============================================================
// Init
// ============================================================

void Class_Sorting_Vision_UART::Init(
    UART_HandleTypeDef *__huart,
    const float &__Coordinate_Scale_MM)
{
    UART_Handler =
        __huart;


    Coordinate_Scale_MM =
        (__Coordinate_Scale_MM > 0.0f)
        ?
        __Coordinate_Scale_MM
        :
        SORTING_UART_DEFAULT_COORDINATE_SCALE_MM;


    Instance =
        this;


    Reset_Parser();


    std::memset(
        Tx_Frame_Buffer,
        0,
        sizeof(Tx_Frame_Buffer)
    );

    std::memset(Tx_Queue, 0, sizeof(Tx_Queue));
    Tx_Queue_Head = 0U;
    Tx_Queue_Tail = 0U;
    Tx_Queue_Count = 0U;

    Stop_Command_Latched = false;
    Stop_Command_Data = Struct_Sorting_Vision_UART_Data{};


    Last_Rx_Data =
        Struct_Sorting_Vision_UART_Data{};


    Queue_Head =
        0U;

    Queue_Tail =
        0U;

    Queue_Count =
        0U;


    Rx_Frame_Count =
        0U;

    Rx_Error_Count =
        0U;

    Rx_Drop_Count =
        0U;


    Alive_Flag =
        0U;

    Alive_Pre_Flag =
        0U;

    Last_Valid_Rx_Tick = HAL_GetTick();


    UART_Status =
        Sorting_Vision_UART_Status_DISABLE;


    if (UART_Handler != nullptr)
    {
        UART_Init(
            UART_Handler,
            Static_UART_Callback,
            SORTING_UART_RX_DMA_LENGTH
        );
    }
}


// ============================================================
// Static callback
// ============================================================

void Class_Sorting_Vision_UART::
Static_UART_Callback(
    uint8_t *Buffer,
    uint16_t Length)
{
    if (Instance != nullptr)
    {
        Instance->
            UART_RxCpltCallback(
                Buffer,
                Length
            );
    }
}


// ============================================================
// RX callback
// ============================================================

void Class_Sorting_Vision_UART::
UART_RxCpltCallback(
    uint8_t *Buffer,
    uint16_t Length)
{
    if (
        Buffer == nullptr
        ||
        Length == 0U
    )
    {
        return;
    }


    for (
        uint16_t i = 0U;
        i < Length;
        ++i
    )
    {
        Parse_Byte(
            Buffer[i]
        );
    }
}


// ============================================================
// Parse byte
// ============================================================

void Class_Sorting_Vision_UART::
Parse_Byte(
    const uint8_t &Byte)
{
    // 等待A5

    if (Rx_Frame_Index == 0U)
    {
        if (
            Byte ==
            SORTING_UART_FRAME_HEAD_0
        )
        {
            Rx_Frame_Buffer[0] =
                Byte;

            Rx_Frame_Index =
                1U;
        }

        return;
    }


    // 等待5A

    if (Rx_Frame_Index == 1U)
    {
        if (
            Byte ==
            SORTING_UART_FRAME_HEAD_1
        )
        {
            Rx_Frame_Buffer[1] =
                Byte;

            Rx_Frame_Index =
                2U;
        }
        else if (
            Byte ==
            SORTING_UART_FRAME_HEAD_0
        )
        {
            Rx_Frame_Buffer[0] =
                Byte;

            Rx_Frame_Index =
                1U;
        }
        else
        {
            Reset_Parser();
        }

        return;
    }


    Rx_Frame_Buffer[
        Rx_Frame_Index++
    ] = Byte;


    if (
        Rx_Frame_Index <
        SORTING_UART_FRAME_LENGTH
    )
    {
        return;
    }


    // ========================================================
    // 完整帧
    // ========================================================

    if (
        Rx_Frame_Buffer[8] ==
            SORTING_UART_FRAME_TAIL_0
        &&
        Rx_Frame_Buffer[9] ==
            SORTING_UART_FRAME_TAIL_1
    )
    {
        Process_Frame(
            Rx_Frame_Buffer
        );

        Reset_Parser();

        return;
    }


    // ========================================================
    // 错误帧
    // ========================================================

    Rx_Error_Count++;


    // 不只保留最后一个字节。错误10字节窗口内部可能已经包含
    // 下一帧的A5 5A帧头；滑动保留该后缀，避免合法帧被整帧吞掉。
    Resync_Parser_After_Error();
}


// ============================================================
// Process frame
// ============================================================

void Class_Sorting_Vision_UART::
Process_Frame(
    const uint8_t *Frame)
{
    if (Frame == nullptr)
    {
        return;
    }


    const auto mode =
        static_cast<
            Enum_Sorting_Vision_Mode
        >(
            Frame[2]
        );


    const uint8_t box_id =
        Frame[7];


    if (
        !Validate_Mode(mode)
        ||
        !Validate_Box_ID(
            mode,
            box_id
        )
    )
    {
        Rx_Error_Count++;

        return;
    }


    const uint16_t x_u16 =
        static_cast<uint16_t>(
            Frame[3]
        )
        |
        (
            static_cast<uint16_t>(
                Frame[4]
            )
            <<
            8U
        );


    const uint16_t y_u16 =
        static_cast<uint16_t>(
            Frame[5]
        )
        |
        (
            static_cast<uint16_t>(
                Frame[6]
            )
            <<
            8U
        );


    Struct_Sorting_Vision_UART_Data
        data{};


    data.Mode =
        mode;


    data.X_Raw =
        static_cast<int16_t>(
            x_u16
        );

    data.Y_Raw =
        static_cast<int16_t>(
            y_u16
        );


    data.X_mm =
        Coordinate_Raw_To_MM(
            data.X_Raw
        );

    data.Y_mm =
        Coordinate_Raw_To_MM(
            data.Y_Raw
        );


    data.Box_ID =
        box_id;

    data.Yaw_Raw =
        static_cast<int8_t>(box_id);

    data.Yaw_Branch =
        0U;

    if (mode == Sorting_Vision_Mode_PICK_TARGET ||
        mode == Sorting_Vision_Mode_MCU_PICK_TARGET_ACK)
    {
        data.Grasp_Axis_Deg =
            Yaw_Raw_To_Deg(data.Yaw_Raw);
        data.Box_ID = 0U;
    }
    else if (mode == Sorting_Vision_Mode_PICK_YAW_BRANCH ||
             mode == Sorting_Vision_Mode_MCU_PICK_YAW_BRANCH_ACK)
    {
        data.Yaw_Branch =
            box_id;
        data.Box_ID = 0U;
    }


    Rx_Frame_Count++;


    data.Sequence =
        Rx_Frame_Count;


    Last_Rx_Data =
        data;

    Last_Valid_Rx_Tick = HAL_GetTick();


    Push_Command(
        data
    );


    Alive_Flag++;


    UART_Status =
        Sorting_Vision_UART_Status_ENABLE;
}


// ============================================================
// Validate mode
// ============================================================

bool Class_Sorting_Vision_UART::
Validate_Mode(
    const Enum_Sorting_Vision_Mode &Mode)
    const
{
    switch (Mode)
    {
        case Sorting_Vision_Mode_HEARTBEAT:

        case Sorting_Vision_Mode_PICK_TARGET:

        case Sorting_Vision_Mode_SORT_RESULT:

        case Sorting_Vision_Mode_CALIB_MOVE:

        case Sorting_Vision_Mode_CALIB_YAW:

        case Sorting_Vision_Mode_STORAGE_TEST:

        case Sorting_Vision_Mode_RETRY_PICK:

        case Sorting_Vision_Mode_PICK_PREOPEN:

        case Sorting_Vision_Mode_PICK_YAW_BRANCH:

        case Sorting_Vision_Mode_STOP:


        case Sorting_Vision_Mode_MCU_HEARTBEAT_ACK:

        case Sorting_Vision_Mode_MCU_PICK_TARGET_ACK:

        case Sorting_Vision_Mode_MCU_RECOGNITION_READY:

        case Sorting_Vision_Mode_MCU_SORT_RESULT_ACK:

        case Sorting_Vision_Mode_MCU_SORT_DONE:

        case Sorting_Vision_Mode_MCU_STOP_ACK:

        case Sorting_Vision_Mode_MCU_BUSY:

        case Sorting_Vision_Mode_MCU_START_WORK:

        case Sorting_Vision_Mode_MCU_STOP_WORK:

        case Sorting_Vision_Mode_MCU_CALIB_MOVE_ACK:

        case Sorting_Vision_Mode_MCU_CALIB_REACHED:

        case Sorting_Vision_Mode_MCU_CALIB_YAW_ACK:

        case Sorting_Vision_Mode_MCU_CALIB_YAW_REACHED:

        case Sorting_Vision_Mode_MCU_STORAGE_TEST_ACK:

        case Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE:

        case Sorting_Vision_Mode_MCU_RETRY_PICK_READY:

        case Sorting_Vision_Mode_MCU_PICK_PREOPEN_ACK:

        case Sorting_Vision_Mode_MCU_PICK_YAW_BRANCH_ACK:

        case Sorting_Vision_Mode_MCU_ERROR:
        {
            return true;
        }


        default:
        {
            return false;
        }
    }
}


// ============================================================
// Validate Box ID
// ============================================================

bool Class_Sorting_Vision_UART::
Validate_Box_ID(
    const Enum_Sorting_Vision_Mode &Mode,
    const uint8_t &Box_ID)
    const
{
    switch (Mode)
    {
        case Sorting_Vision_Mode_SORT_RESULT:

        case Sorting_Vision_Mode_MCU_SORT_RESULT_ACK:

        case Sorting_Vision_Mode_MCU_SORT_DONE:

        case Sorting_Vision_Mode_STORAGE_TEST:

        case Sorting_Vision_Mode_MCU_STORAGE_TEST_ACK:

        case Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE:
        {
            return
                Box_ID >= 1U
                &&
                Box_ID <= 6U;
        }


        case Sorting_Vision_Mode_PICK_TARGET:
        case Sorting_Vision_Mode_MCU_PICK_TARGET_ACK:
        {
            const int8_t yaw_raw = static_cast<int8_t>(Box_ID);
            return yaw_raw >= -90 && yaw_raw <= 90;
        }

        case Sorting_Vision_Mode_PICK_YAW_BRANCH:
        case Sorting_Vision_Mode_MCU_PICK_YAW_BRANCH_ACK:
        {
            return Box_ID <= 1U;
        }

        case Sorting_Vision_Mode_MCU_ERROR:
        {
            return Box_ID <= Sorting_Vision_Error_Grap_TOF_Timeout;
        }

        default:
        {
            return Box_ID == 0U;
        }
    }
}


// ============================================================
// Queue push
// ============================================================

void Class_Sorting_Vision_UART::
Push_Command(
    const Struct_Sorting_Vision_UART_Data &Data_In)
{
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();

    // STOP是安全事件，使用独立锁存并清掉尚未执行的普通命令。
    // 这样即使普通队列被大量心跳/命令填满，STOP也不会被淘汰。
    if (Data_In.Mode == Sorting_Vision_Mode_STOP)
    {
        Rx_Drop_Count += Queue_Count;
        Queue_Head = 0U;
        Queue_Tail = 0U;
        Queue_Count = 0U;

        Stop_Command_Data = Data_In;
        Stop_Command_Latched = true;

        if (primask == 0U) __enable_irq();
        return;
    }

    if (Queue_Count >= SORTING_UART_COMMAND_QUEUE_LENGTH)
    {
        Queue_Head = static_cast<uint8_t>(
            (Queue_Head + 1U) % SORTING_UART_COMMAND_QUEUE_LENGTH);
        Queue_Count--;
        Rx_Drop_Count++;
    }

    Command_Queue[Queue_Tail] = Data_In;
    Queue_Tail = static_cast<uint8_t>(
        (Queue_Tail + 1U) % SORTING_UART_COMMAND_QUEUE_LENGTH);
    Queue_Count++;

    if (primask == 0U) __enable_irq();
}


// ============================================================
// Pop command
// ============================================================

bool Class_Sorting_Vision_UART::
Pop_Command(
    Struct_Sorting_Vision_UART_Data &Data_Out)
{
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();

    // STOP永远先于普通命令交给握手层。
    if (Stop_Command_Latched)
    {
        Data_Out = Stop_Command_Data;
        Stop_Command_Latched = false;
        Stop_Command_Data = Struct_Sorting_Vision_UART_Data{};
        if (primask == 0U) __enable_irq();
        return true;
    }

    if (Queue_Count == 0U)
    {
        if (primask == 0U) __enable_irq();
        return false;
    }

    Data_Out = Command_Queue[Queue_Head];
    Queue_Head = static_cast<uint8_t>(
        (Queue_Head + 1U) % SORTING_UART_COMMAND_QUEUE_LENGTH);
    Queue_Count--;

    if (primask == 0U) __enable_irq();
    return true;
}


void Class_Sorting_Vision_UART::Clear_Command_Queue()
{
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();

    Rx_Drop_Count += Queue_Count;
    Queue_Head = 0U;
    Queue_Tail = 0U;
    Queue_Count = 0U;

    if (primask == 0U) __enable_irq();
}


// ============================================================
// Send frame
//
// 返回HAL_OK表示“帧已经进入可靠发送队列”，不要求DMA此刻空闲。
// DMA真正空闲时由Service_Tx()发送，避免同一周期连续ACK/READY/DONE丢帧。
// ============================================================

HAL_StatusTypeDef
Class_Sorting_Vision_UART::
Send_Frame(
    const Enum_Sorting_Vision_Mode &Mode,
    const float &X_mm,
    const float &Y_mm,
    const uint8_t &Box_ID)
{
    if (UART_Handler == nullptr)
    {
        return HAL_ERROR;
    }

    if (!Validate_Mode(Mode) || !Validate_Box_ID(Mode, Box_ID))
    {
        return HAL_ERROR;
    }

    const int16_t x_raw = Coordinate_MM_To_Raw(X_mm);
    const int16_t y_raw = Coordinate_MM_To_Raw(Y_mm);
    const uint16_t x_u16 = static_cast<uint16_t>(x_raw);
    const uint16_t y_u16 = static_cast<uint16_t>(y_raw);

    uint8_t frame[SORTING_UART_FRAME_LENGTH] = {0};
    frame[0] = SORTING_UART_FRAME_HEAD_0;
    frame[1] = SORTING_UART_FRAME_HEAD_1;
    frame[2] = static_cast<uint8_t>(Mode);
    frame[3] = static_cast<uint8_t>(x_u16 & 0x00FFU);
    frame[4] = static_cast<uint8_t>((x_u16 >> 8U) & 0x00FFU);
    frame[5] = static_cast<uint8_t>(y_u16 & 0x00FFU);
    frame[6] = static_cast<uint8_t>((y_u16 >> 8U) & 0x00FFU);
    frame[7] = Box_ID;
    frame[8] = SORTING_UART_FRAME_TAIL_0;
    frame[9] = SORTING_UART_FRAME_TAIL_1;

    const uint32_t primask = __get_PRIMASK();
    __disable_irq();

    if (Tx_Queue_Count >= SORTING_UART_TX_QUEUE_LENGTH)
    {
        if (primask == 0U) __enable_irq();
        return HAL_BUSY;
    }

    std::memcpy(
        Tx_Queue[Tx_Queue_Tail],
        frame,
        SORTING_UART_FRAME_LENGTH);
    Tx_Queue_Tail = static_cast<uint8_t>(
        (Tx_Queue_Tail + 1U) % SORTING_UART_TX_QUEUE_LENGTH);
    Tx_Queue_Count++;

    if (primask == 0U) __enable_irq();

    Service_Tx();
    return HAL_OK;
}


void Class_Sorting_Vision_UART::Service_Tx()
{
    if (UART_Handler == nullptr)
    {
        return;
    }

    const uint32_t primask = __get_PRIMASK();
    __disable_irq();

    if (Tx_Queue_Count == 0U ||
        UART_Handler->gState != HAL_UART_STATE_READY)
    {
        if (primask == 0U) __enable_irq();
        return;
    }

    std::memcpy(
        Tx_Frame_Buffer,
        Tx_Queue[Tx_Queue_Head],
        SORTING_UART_FRAME_LENGTH);

    const HAL_StatusTypeDef status = static_cast<HAL_StatusTypeDef>(
        UART_Transmit_Data(
            UART_Handler,
            Tx_Frame_Buffer,
            SORTING_UART_FRAME_LENGTH));

    if (status == HAL_OK)
    {
        Tx_Queue_Head = static_cast<uint8_t>(
            (Tx_Queue_Head + 1U) % SORTING_UART_TX_QUEUE_LENGTH);
        Tx_Queue_Count--;
    }

    if (primask == 0U) __enable_irq();
}


// ============================================================
// mm -> raw
// ============================================================

int16_t
Class_Sorting_Vision_UART::
Coordinate_MM_To_Raw(
    const float &Coordinate_mm)
    const
{
    const float raw_f =
        Coordinate_mm
        /
        Coordinate_Scale_MM;


    if (
        raw_f >=
        static_cast<float>(
            INT16_MAX
        )
    )
    {
        return INT16_MAX;
    }


    if (
        raw_f <=
        static_cast<float>(
            INT16_MIN
        )
    )
    {
        return INT16_MIN;
    }


    return
        static_cast<int16_t>(
            std::lround(
                raw_f
            )
        );
}


// ============================================================
// raw -> mm
// ============================================================

float
Class_Sorting_Vision_UART::
Coordinate_Raw_To_MM(
    const int16_t &Coordinate_Raw)
    const
{
    return
        static_cast<float>(
            Coordinate_Raw
        )
        *
        Coordinate_Scale_MM;
}


// ============================================================
// Bad-frame sliding resynchronization
// ============================================================

void Class_Sorting_Vision_UART::Resync_Parser_After_Error()
{
    uint8_t header_index = 0xFFU;

    for (uint8_t i = 1U; i + 1U < SORTING_UART_FRAME_LENGTH; ++i)
    {
        if (Rx_Frame_Buffer[i] == SORTING_UART_FRAME_HEAD_0 &&
            Rx_Frame_Buffer[i + 1U] == SORTING_UART_FRAME_HEAD_1)
        {
            header_index = i;
            break;
        }
    }

    if (header_index != 0xFFU)
    {
        const uint8_t keep = static_cast<uint8_t>(
            SORTING_UART_FRAME_LENGTH - header_index);
        std::memmove(
            Rx_Frame_Buffer,
            &Rx_Frame_Buffer[header_index],
            keep);
        std::memset(
            &Rx_Frame_Buffer[keep],
            0,
            SORTING_UART_FRAME_LENGTH - keep);
        Rx_Frame_Index = keep;
        return;
    }

    const bool keep_a5 =
        Rx_Frame_Buffer[SORTING_UART_FRAME_LENGTH - 1U] ==
        SORTING_UART_FRAME_HEAD_0;

    Reset_Parser();
    if (keep_a5)
    {
        Rx_Frame_Buffer[0] = SORTING_UART_FRAME_HEAD_0;
        Rx_Frame_Index = 1U;
    }
}


// ============================================================
// Reset parser
// ============================================================

void Class_Sorting_Vision_UART::
Reset_Parser()
{
    Rx_Frame_Index =
        0U;


    std::memset(
        Rx_Frame_Buffer,
        0,
        sizeof(Rx_Frame_Buffer)
    );
}


// ============================================================
// Alive
// ============================================================

void Class_Sorting_Vision_UART::
TIM_1000ms_Alive_PeriodElapsedCallback()
{
    UART_Status =
        (
            Alive_Flag ==
            Alive_Pre_Flag
        )
        ?
        Sorting_Vision_UART_Status_DISABLE
        :
        Sorting_Vision_UART_Status_ENABLE;


    Alive_Pre_Flag =
        Alive_Flag;
}

// ============================================================
// PICK yaw deg <-> signed int8
// ============================================================
int8_t Class_Sorting_Vision_UART::Yaw_Deg_To_Raw(const float &Yaw_Deg)
{
    float value = Yaw_Deg;
    if (value > 90.0f) value = 90.0f;
    if (value < -90.0f) value = -90.0f;
    return static_cast<int8_t>(std::lround(value));
}

float Class_Sorting_Vision_UART::Yaw_Raw_To_Deg(const int8_t &Yaw_Raw)
{
    return static_cast<float>(Yaw_Raw);
}
