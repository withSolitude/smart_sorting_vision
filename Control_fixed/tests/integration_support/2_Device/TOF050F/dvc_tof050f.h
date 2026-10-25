#ifndef TEST_TOF_H
#define TEST_TOF_H
#include "test_hardware.h"
struct Struct_Test_TOF_Data { uint32_t Rx_Count=0; };
class Class_TOF050F
{
public:
    bool Valid=true;
    uint16_t Distance=30;
    Struct_Test_TOF_Data Data;
    bool Get_Distance_Valid() const { return Valid; }
    uint16_t Get_Distance_mm() const { return Distance; }
    const Struct_Test_TOF_Data &Get_Data() const { return Data; }
    void TIM_100ms_PeriodElapsedCallback() {}
};
#endif
