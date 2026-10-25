#ifndef TEST_XY_MOTION_H
#define TEST_XY_MOTION_H
#include "test_hardware.h"
#include "3_Robot/xy_motion_config.h"
class Class_XY_Motion
{
public:
    struct Move { float X,Y,VX,VY; uint32_t Tick; };
    bool Position_Valid=true, Fault=false, Motion_Finished=true, Stop_Confirmed=true;
    // 当前机构以上电位置作为软件原点，待机点的 XY 夹具也从 (0,0) 开始。
    float X=0,Y=0;
    uint32_t Stop_Calls=0;
    HAL_StatusTypeDef Move_Status=HAL_OK;
    HAL_StatusTypeDef Homing_Status=HAL_ERROR;
    std::vector<Move> Moves;
    void Init() {}
    void TIM_1ms_PeriodElapsedCallback() {}
    bool Get_Position_Valid() const { return Position_Valid; }
    bool Get_Fault() const { return Fault; }
    bool Get_Motion_Finished() const { return Motion_Finished; }
    bool Get_Stop_Confirmed() const { return Stop_Confirmed; }
    bool Get_Homing_Active() const { return false; }
    float Get_X_mm() const { return X; }
    float Get_Y_mm() const { return Y; }
    void Stop() { ++Stop_Calls; Motion_Finished=false; Stop_Confirmed=false; }
    HAL_StatusTypeDef Start_Collision_Homing() { return Homing_Status; }
    void Stop_Collision_Homing() { ++Stop_Calls; }
    HAL_StatusTypeDef Move_To_mm(float x,float y,float vx,float vy)
    {
        if(Move_Status==HAL_OK) { Moves.push_back({x,y,vx,vy,HAL_GetTick()}); Motion_Finished=false; Stop_Confirmed=false; }
        return Move_Status;
    }
    HAL_StatusTypeDef Confirm_Origin_mm(float x,float y)
    { X=x;Y=y;Position_Valid=true;Fault=false;Motion_Finished=true;Stop_Confirmed=true; return HAL_OK; }
    void Arrive() { if(!Moves.empty()) { X=Moves.back().X;Y=Moves.back().Y; } Motion_Finished=true;Stop_Confirmed=true; }
};
#endif
