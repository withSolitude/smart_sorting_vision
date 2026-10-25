#ifndef EMM_V5_HAL_STUB_H
#define EMM_V5_HAL_STUB_H
#include <cstdint>

enum HAL_StatusTypeDef { HAL_OK = 0, HAL_ERROR = 1, HAL_BUSY = 2, HAL_TIMEOUT = 3 };
struct FDCAN_HandleTypeDef { unsigned Instance = 0; };
struct FDCAN_RxHeaderTypeDef {
    uint32_t Identifier = 0;
    uint32_t IdType = 0;
    uint32_t RxFrameType = 0;
    uint32_t DataLength = 0;
    uint32_t FDFormat = 0;
};
constexpr uint32_t FDCAN_STANDARD_ID = 0;
constexpr uint32_t FDCAN_EXTENDED_ID = 1;
constexpr uint32_t FDCAN_DATA_FRAME = 0;
constexpr uint32_t FDCAN_REMOTE_FRAME = 1;
constexpr uint32_t FDCAN_CLASSIC_CAN = 0;
constexpr uint32_t FDCAN_FD_CAN = 1;
constexpr uint32_t FDCAN_DLC_BYTES_4 = 4;
constexpr uint32_t FDCAN_DLC_BYTES_7 = 7;
constexpr uint32_t FDCAN_DLC_BYTES_8 = 8;
extern FDCAN_HandleTypeDef hfdcan1, hfdcan2, hfdcan3;
uint32_t HAL_GetTick();
uint32_t HAL_FDCAN_GetTxFifoFreeLevel(FDCAN_HandleTypeDef *);
HAL_StatusTypeDef CAN_Transmit_Extended_Data(FDCAN_HandleTypeDef *, uint32_t,
                                           const uint8_t *, uint8_t);
inline uint32_t __get_PRIMASK() { return 0; }
inline void __disable_irq() {}
inline void __set_PRIMASK(uint32_t) {}
#endif
