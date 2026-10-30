#ifndef DVC_SORTING_VISION_UART_H
#define DVC_SORTING_VISION_UART_H

#include <cstdint>

#include "1_Middleware/1_Driver/UART/drv_uart.h"


// ============================================================
// 固定10字节协议
//
// [0] A5
// [1] 5A
// [2] MODE
// [3] X_L
// [4] X_H
// [5] Y_L
// [6] Y_H
// [7] AUX
//     PICK_TARGET / MCU_PICK_TARGET_ACK: signed int8 抓取闭合轴角度(deg)
//     SORT_RESULT / SORT_ACK / SORT_DONE: Box_ID
//     MCU_ERROR: Error_Code
// [8] 0D
// [9] 0A
//
// X/Y:
// int16_t
// 1 LSB = 0.1 mm
//
// PICK yaw:
// signed int8_t, 1 LSB = 1 deg, 合法范围 -90~+90 deg。
// 含义是机械XY坐标系中的“夹爪闭合轴方向”，不是舵机原始角度。
// ============================================================

#define SORTING_UART_FRAME_HEAD_0                 0xA5U
#define SORTING_UART_FRAME_HEAD_1                 0x5AU

#define SORTING_UART_FRAME_TAIL_0                 0x0DU
#define SORTING_UART_FRAME_TAIL_1                 0x0AU

#define SORTING_UART_FRAME_LENGTH                 10U
#define SORTING_UART_RX_DMA_LENGTH                64U

#define SORTING_UART_COMMAND_QUEUE_LENGTH         8U
#define SORTING_UART_TX_QUEUE_LENGTH              16U

#define SORTING_UART_DEFAULT_COORDINATE_SCALE_MM  0.1f


// ============================================================
// 通信模式
// ============================================================

enum Enum_Sorting_Vision_Mode : uint8_t
{
    // ========================================================
    // Jetson -> STM32
    // ========================================================

    // 心跳
    Sorting_Vision_Mode_HEARTBEAT =
        0x00,

    // 顶部相机识别完成
    //
    // X/Y有效
    // Box_ID = 0
    Sorting_Vision_Mode_PICK_TARGET =
        0x01,

    // 侧面相机识别完成
    //
    // Box_ID = 1~6
    Sorting_Vision_Mode_SORT_RESULT =
        0x02,

    // XY标定：仅移动X/Y，不执行抓取流程
    Sorting_Vision_Mode_CALIB_MOVE =
        0x03,

    // Yaw标定：仅直接控制Yaw舵机，不下降、不夹取。
    // 协议复用X字段传舵机命令角度，单位deg，0.1deg/LSB；Y/AUX=0。
    Sorting_Vision_Mode_CALIB_YAW =
        0x04,

    // 储物机构独立测试：
    // 仅允许整机STOPPED时执行，不经过顶部抓取/侧面正式握手。
    // AUX = Box_ID(1~6)
    Sorting_Vision_Mode_STORAGE_TEST =
        0x05,

    // 识别托盘连续确认无物体：放弃本次失败夹取，但不停止整场。
    // 下位机收到后只做 Recognition -> Disable 安全回位，
    // 完成后回到 WAIT_PICK_TARGET 等待下一次顶部抓取。
    Sorting_Vision_Mode_RETRY_PICK =
        0x06,

    // 动态夹爪预开口参数。X字段为目标预开口(mm)，Y/AUX=0。
    // 只锁存参数，不立即动作；真正预收缩仍由抓取状态机执行。
    Sorting_Vision_Mode_PICK_PREOPEN =
        0x07,

    // 上位机主动急停/取消
    //
    // 平时主要由STM32实体按键控制停机，
    // 这个命令仍保留。
    Sorting_Vision_Mode_STOP =
        0x7F,


    // ========================================================
    // STM32 -> Jetson
    // ========================================================

    Sorting_Vision_Mode_MCU_HEARTBEAT_ACK =
        0x80,

    // STM32已经收到并锁定X/Y
    Sorting_Vision_Mode_MCU_PICK_TARGET_ACK =
        0x81,

    // 第一阶段机械动作完成
    //
    // 物体已经被放到固定侧面识别托盘
    // Jetson现在开始侧面相机识别
    Sorting_Vision_Mode_MCU_RECOGNITION_READY =
        0x82,

    // STM32已经收到Box_ID
    Sorting_Vision_Mode_MCU_SORT_RESULT_ACK =
        0x83,

    // 一件物体真正完成投放
    //
    // Jetson收到以后：
    // 1. GUI成功数量+1
    // 2. 开始下一次顶部识别
    Sorting_Vision_Mode_MCU_SORT_DONE =
        0x84,

    // 上位机发送STOP后的ACK
    Sorting_Vision_Mode_MCU_STOP_ACK =
        0x85,

    Sorting_Vision_Mode_MCU_BUSY =
        0x86,

    // ========================================================
    // 新增：
    // STM32实体按键启动
    // ========================================================

    Sorting_Vision_Mode_MCU_START_WORK =
        0x87,

    // ========================================================
    // 新增：
    // STM32实体按键停止
    // ========================================================

    Sorting_Vision_Mode_MCU_STOP_WORK =
        0x88,

    // XY标定命令已接收
    Sorting_Vision_Mode_MCU_CALIB_MOVE_ACK =
        0x89,

    // XY标定目标已真正到位并稳定
    Sorting_Vision_Mode_MCU_CALIB_REACHED =
        0x8A,

    // Yaw标定命令已接收
    Sorting_Vision_Mode_MCU_CALIB_YAW_ACK =
        0x8B,

    // Yaw标定命令的设定运动时间已经完成。
    // 注意：当前众灵舵机未周期回读角度，因此这是“时间到位”，不是闭环反馈到位。
    Sorting_Vision_Mode_MCU_CALIB_YAW_REACHED =
        0x8C,

    // 储物机构独立测试 ACK / DONE
    // AUX = Box_ID(1~6)
    Sorting_Vision_Mode_MCU_STORAGE_TEST_ACK =
        0x8D,

    Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE =
        0x8E,

    // RETRY_PICK 恢复完成：夹爪已经真正回到 Disable，
    // 握手状态已经重新进入 WAIT_PICK_TARGET。
    Sorting_Vision_Mode_MCU_RETRY_PICK_READY =
        0x8F,

    // 动态预开口参数已锁存，X字段回显预开口(mm)。
    Sorting_Vision_Mode_MCU_PICK_PREOPEN_ACK =
        0x90,

    // 错误
    Sorting_Vision_Mode_MCU_ERROR =
        0xE0,
};


// ============================================================
// MCU_ERROR的AUX错误码
// ============================================================

enum Enum_Sorting_Vision_Error_Code : uint8_t
{
    Sorting_Vision_Error_None = 0U,

    // 兼容既有错误码
    Sorting_Vision_Error_Grap_Generic = 1U,
    Sorting_Vision_Error_Storage_Request = 2U,
    Sorting_Vision_Error_Storage_Runtime = 3U,
    Sorting_Vision_Error_Reserved = 4U,
    Sorting_Vision_Error_Calib_Move = 5U,
    Sorting_Vision_Error_Calib_Yaw = 6U,

    // 抓取流程细分错误码
    Sorting_Vision_Error_Grap_Start_Timeout = 7U,
    Sorting_Vision_Error_Grap_XY_Position_Invalid = 8U,
    Sorting_Vision_Error_Grap_TOF_Invalid = 9U,
    Sorting_Vision_Error_Grap_Target_Invalid = 10U,
    Sorting_Vision_Error_Grap_TOF_Timeout = 11U,
};


// ============================================================
// UART在线状态
// ============================================================

enum Enum_Sorting_Vision_UART_Status : uint8_t
{
    Sorting_Vision_UART_Status_DISABLE = 0,

    Sorting_Vision_UART_Status_ENABLE,
};


// ============================================================
// 接收到的一帧解析结果
// ============================================================

struct Struct_Sorting_Vision_UART_Data
{
    Enum_Sorting_Vision_Mode Mode =
        Sorting_Vision_Mode_HEARTBEAT;

    int16_t X_Raw =
        0;

    int16_t Y_Raw =
        0;

    float X_mm =
        0.0f;

    float Y_mm =
        0.0f;

    uint8_t Box_ID =
        0;

    int8_t Yaw_Raw =
        0;

    float Grasp_Axis_Deg =
        0.0f;

    uint32_t Sequence =
        0;
};


// ============================================================
// UART通信类
// ============================================================

class Class_Sorting_Vision_UART
{
public:

    // ========================================================
    // 初始化
    // ========================================================

    void Init(
        UART_HandleTypeDef *__huart,

        const float &__Coordinate_Scale_MM =
            SORTING_UART_DEFAULT_COORDINATE_SCALE_MM
    );


    // ========================================================
    // UART接收
    // ========================================================

    void UART_RxCpltCallback(
        uint8_t *Buffer,
        uint16_t Length
    );


    // ========================================================
    // 从接收队列取命令
    // ========================================================

    bool Pop_Command(
        Struct_Sorting_Vision_UART_Data &Data_Out
    );

    // 清空尚未消费的普通命令。STOP使用独立锁存，不会被本函数清掉。
    void Clear_Command_Queue();

    // 推进DMA发送队列。可在1ms周期中调用，Send_Frame也会主动调用一次。
    void Service_Tx();


    // ========================================================
    // 通用发送
    // ========================================================

    HAL_StatusTypeDef Send_Frame(
        const Enum_Sorting_Vision_Mode &Mode,

        const float &X_mm =
            0.0f,

        const float &Y_mm =
            0.0f,

        const uint8_t &Box_ID =
            0
    );


    // ========================================================
    // Heartbeat ACK
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Heartbeat_Ack()
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_HEARTBEAT_ACK
        );
    }


    // ========================================================
    // 动态预开口参数 ACK
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Pick_Preopen_Ack(const float &Preopen_mm)
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_PICK_PREOPEN_ACK,
            Preopen_mm,
            0.0f,
            0
        );
    }


    // ========================================================
    // 顶部坐标ACK
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Pick_Target_Ack(
        const float &X_mm,
        const float &Y_mm,
        const float &Grasp_Axis_Deg
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_PICK_TARGET_ACK,
            X_mm,
            Y_mm,
            static_cast<uint8_t>(Yaw_Deg_To_Raw(Grasp_Axis_Deg))
        );
    }


    // ========================================================
    // 通知Jetson：
    // 物体已经到侧面识别台
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Recognition_Ready(
        const float &X_mm =
            0.0f,

        const float &Y_mm =
            0.0f
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_RECOGNITION_READY,
            X_mm,
            Y_mm,
            0
        );
    }


    // ========================================================
    // Box ID ACK
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Sort_Result_Ack(
        const uint8_t &Box_ID
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_SORT_RESULT_ACK,
            0.0f,
            0.0f,
            Box_ID
        );
    }


    // ========================================================
    // 一件物体真正完成
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Sort_Done(
        const uint8_t &Box_ID
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_SORT_DONE,
            0.0f,
            0.0f,
            Box_ID
        );
    }


    // ========================================================
    // PC STOP ACK
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Stop_Ack()
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_STOP_ACK
        );
    }


    // ========================================================
    // Busy
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Busy()
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_BUSY
        );
    }


    // ========================================================
    // STM32实体按键启动
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Start_Work()
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_START_WORK
        );
    }


    // ========================================================
    // STM32实体按键停止
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Stop_Work()
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_STOP_WORK
        );
    }




    inline HAL_StatusTypeDef
    Send_Calib_Move_Ack(const float &X_mm, const float &Y_mm)
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_CALIB_MOVE_ACK,
            X_mm, Y_mm, 0
        );
    }

    inline HAL_StatusTypeDef
    Send_Calib_Reached(const float &X_mm, const float &Y_mm)
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_CALIB_REACHED,
            X_mm, Y_mm, 0
        );
    }

    inline HAL_StatusTypeDef
    Send_Calib_Yaw_Ack(const float &Servo_Command_Deg)
    {
        // CALIB_YAW复用X字段承载角度，Send_Frame仍以0.1/LSB编码。
        return Send_Frame(
            Sorting_Vision_Mode_MCU_CALIB_YAW_ACK,
            Servo_Command_Deg, 0.0f, 0
        );
    }

    inline HAL_StatusTypeDef
    Send_Calib_Yaw_Reached(const float &Servo_Command_Deg)
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_CALIB_YAW_REACHED,
            Servo_Command_Deg, 0.0f, 0
        );
    }


    // ========================================================
    // 储物机构独立测试 ACK / DONE
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Storage_Test_Ack(
        const uint8_t &Box_ID
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_STORAGE_TEST_ACK,
            0.0f,
            0.0f,
            Box_ID
        );
    }

    inline HAL_StatusTypeDef
    Send_Storage_Test_Done(
        const uint8_t &Box_ID
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE,
            0.0f,
            0.0f,
            Box_ID
        );
    }

    // ========================================================
    // 识别台空载恢复完成
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Retry_Pick_Ready()
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_RETRY_PICK_READY
        );
    }

    // ========================================================
    // Error
    // ========================================================

    inline HAL_StatusTypeDef
    Send_Error(
        const uint8_t &Error_Code =
            0
    )
    {
        return Send_Frame(
            Sorting_Vision_Mode_MCU_ERROR,
            0.0f,
            0.0f,
            Error_Code
        );
    }


    // ========================================================
    // 在线检测
    // ========================================================

    void TIM_1000ms_Alive_PeriodElapsedCallback();


    // ========================================================
    // Getter
    // ========================================================

    inline Enum_Sorting_Vision_UART_Status
    Get_Status() const
    {
        return UART_Status;
    }

    // 最近一次成功解析完整视觉帧的时间，用于区分链路掉线和动作超时。
    inline uint32_t Get_Last_Valid_Rx_Tick() const
    {
        return Last_Valid_Rx_Tick;
    }

    inline bool Is_Link_Alive(uint32_t timeout_ms = 3000U) const
    {
        return static_cast<uint32_t>(HAL_GetTick() - Last_Valid_Rx_Tick) <= timeout_ms &&
            UART_Status == Sorting_Vision_UART_Status_ENABLE;
    }

    inline uint32_t
    Get_Rx_Frame_Count() const
    {
        return Rx_Frame_Count;
    }

    inline uint32_t
    Get_Rx_Error_Count() const
    {
        return Rx_Error_Count;
    }

    inline uint32_t
    Get_Rx_Drop_Count() const
    {
        return Rx_Drop_Count;
    }

    inline uint8_t
    Get_Queue_Count() const
    {
        return Queue_Count;
    }

    inline uint8_t
    Get_Tx_Queue_Count() const
    {
        return Tx_Queue_Count;
    }

    inline float
    Get_Coordinate_Scale_MM() const
    {
        return Coordinate_Scale_MM;
    }

    inline const Struct_Sorting_Vision_UART_Data &
    Get_Last_Rx_Data() const
    {
        return Last_Rx_Data;
    }


private:

    static void Static_UART_Callback(
        uint8_t *Buffer,
        uint16_t Length
    );

    void Parse_Byte(
        const uint8_t &Byte
    );

    void Process_Frame(
        const uint8_t *Frame
    );

    void Reset_Parser();
    void Resync_Parser_After_Error();

    void Push_Command(
        const Struct_Sorting_Vision_UART_Data &Data_In
    );

    bool Validate_Mode(
        const Enum_Sorting_Vision_Mode &Mode
    ) const;

    bool Validate_Box_ID(
        const Enum_Sorting_Vision_Mode &Mode,
        const uint8_t &Box_ID
    ) const;

    int16_t Coordinate_MM_To_Raw(
        const float &Coordinate_mm
    ) const;

    float Coordinate_Raw_To_MM(
        const int16_t &Coordinate_Raw
    ) const;

    static int8_t Yaw_Deg_To_Raw(const float &Yaw_Deg);
    static float Yaw_Raw_To_Deg(const int8_t &Yaw_Raw);


    UART_HandleTypeDef *UART_Handler =
        nullptr;

    static Class_Sorting_Vision_UART *Instance;


    float Coordinate_Scale_MM =
        SORTING_UART_DEFAULT_COORDINATE_SCALE_MM;


    uint8_t
    Rx_Frame_Buffer[
        SORTING_UART_FRAME_LENGTH
    ] = {0};

    uint8_t Rx_Frame_Index =
        0;


    // 当前DMA正在读取的稳定缓冲区。只有UART READY时才会覆盖。
    uint8_t
    Tx_Frame_Buffer[
        SORTING_UART_FRAME_LENGTH
    ] = {0};

    // 每个待发帧都有独立副本，避免DMA BUSY时丢ACK/READY/DONE。
    uint8_t
    Tx_Queue[SORTING_UART_TX_QUEUE_LENGTH][SORTING_UART_FRAME_LENGTH] = {{0}};

    volatile uint8_t Tx_Queue_Head = 0;
    volatile uint8_t Tx_Queue_Tail = 0;
    volatile uint8_t Tx_Queue_Count = 0;


    Struct_Sorting_Vision_UART_Data
        Last_Rx_Data{};


    Struct_Sorting_Vision_UART_Data
        Command_Queue[
            SORTING_UART_COMMAND_QUEUE_LENGTH
        ]{};


    volatile uint8_t Queue_Head =
        0;

    volatile uint8_t Queue_Tail =
        0;

    volatile uint8_t Queue_Count =
        0;

    // STOP独立锁存，不能被8帧普通命令队列溢出淘汰。
    volatile bool Stop_Command_Latched = false;
    Struct_Sorting_Vision_UART_Data Stop_Command_Data{};


    volatile uint32_t Rx_Frame_Count =
        0;

    volatile uint32_t Rx_Error_Count =
        0;

    volatile uint32_t Rx_Drop_Count =
        0;


    uint32_t Alive_Flag =
        0;

    uint32_t Alive_Pre_Flag =
        0;

    volatile uint32_t Last_Valid_Rx_Tick =
        0U;


    Enum_Sorting_Vision_UART_Status
        UART_Status =
            Sorting_Vision_UART_Status_DISABLE;
};


#endif
