#include "4_Interaction/ita_robot.h"
#include "test_hardware.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

FDCAN_HandleTypeDef hfdcan1{1}, hfdcan2{2}, hfdcan3{3};
UART_HandleTypeDef huart3,huart4;
static uint32_t tick=0;
uint32_t HAL_GetTick() { return tick; }
void HAL_Delay(uint32_t ms) { tick+=ms; }
static std::deque<Struct_Sorting_Vision_UART_Data> incoming;
static std::vector<Enum_Sorting_Vision_Mode> sent;
static HAL_StatusTypeDef tx_status=HAL_OK;
bool Class_Sorting_Vision_UART::Pop_Command(Struct_Sorting_Vision_UART_Data &data)
{ if(incoming.empty())return false;data=incoming.front();incoming.pop_front();return true; }
void Class_Sorting_Vision_UART::Clear_Command_Queue() { incoming.clear(); }
void Class_Sorting_Vision_UART::Init(UART_HandleTypeDef*,const float&) {}
void Class_Sorting_Vision_UART::Service_Tx() {}
void Class_Sorting_Vision_UART::TIM_1000ms_Alive_PeriodElapsedCallback() {}
HAL_StatusTypeDef Class_Sorting_Vision_UART::Send_Frame(const Enum_Sorting_Vision_Mode &mode,
    const float&,const float&,const uint8_t&)
{ if(tx_status==HAL_OK)sent.push_back(mode);return tx_status; }
int8_t Class_Sorting_Vision_UART::Yaw_Deg_To_Raw(const float &angle)
{ return static_cast<int8_t>(std::round(angle)); }

static void queue(Enum_Sorting_Vision_Mode mode,float x=0,float y=0,uint8_t box=0,float yaw=0)
{ Struct_Sorting_Vision_UART_Data d;d.Mode=mode;d.X_mm=x;d.Y_mm=y;d.Box_ID=box;d.Grasp_Axis_Deg=yaw;incoming.push_back(d); }
static unsigned sent_count(Enum_Sorting_Vision_Mode mode)
{ unsigned n=0;for(auto m:sent)if(m==mode)++n;return n; }
static void step(Class_Robot &r,uint32_t ms,bool arrive=false,bool follow_dm=false)
{
    for(uint32_t i=0;i<ms;++i)
    {
        ++tick;
        if(arrive)r.Holding_Jaw.XY_Motion.Arrive();
        if(follow_dm)r.Sorting_Storage.DM4310.Angle=r.Sorting_Storage.Get_DM4310_Target_Angle();
        if(tick%10U==0U)r.Sorting_Storage.DM4310.Refresh();
        if(tick%20U==0U)++r.Holding_Jaw.TOF050F.Data.Rx_Count;
        r.TIM_1ms_Calculate_Callback();
        if(tick%2U==0U)r.TIM_2ms_Calculate_PeriodElapsedCallback();
        if(tick%10U==0U)r.TIM_10ms_Calculate_PeriodElapsedCallback();
        if(tick%100U==0U) {r.TIM_100ms_Calculate_Callback();r.TIM_100ms_Alive_PeriodElapsedCallback();}
    }
}
static Class_Robot ready_robot()
{
    tick=0;incoming.clear();sent.clear();tx_status=HAL_OK;
    huart3.gState=huart4.gState=HAL_UART_STATE_READY;
    Class_Robot r;r.Init();step(r,1100);
    assert(r.Holding_Jaw.Get_Disable_Ready()&&r.Sorting_Storage.Get_Disable_Ready());
    return r;
}

static void test_calibration_start_returns_home()
{
    auto r=ready_robot();queue(Sorting_Vision_Mode_CALIB_MOVE,-10,10);
    step(r,500);
    assert(r.Get_Calibration_Active());
    assert(sent_count(Sorting_Vision_Mode_MCU_CALIB_REACHED)==0U);
    r.Holding_Jaw.XY_Motion.Arrive();step(r,500);
    assert(!r.Get_Calibration_Active());
    assert(!r.Holding_Jaw.Get_Disable_Ready());
    assert(r.Request_Sorting_Start());step(r,1100);
    assert(!r.Get_Sorting_Running());
    assert(sent_count(Sorting_Vision_Mode_MCU_START_WORK)==0U);
    r.Holding_Jaw.XY_Motion.Arrive();step(r,3000);
    assert(r.Holding_Jaw.Get_Disable_Ready()&&r.Get_Sorting_Running());
    assert(sent_count(Sorting_Vision_Mode_MCU_START_WORK)==1U);
    std::puts("PASS: robot calibration reaches by feedback and subsequent START waits for actual standby return");
}

static void test_fault_stopped_ack_and_origin_recovery()
{
    auto r=ready_robot();
    r.Holding_Jaw.XY_Motion.Fault=true;r.Holding_Jaw.XY_Motion.Position_Valid=false;
    queue(Sorting_Vision_Mode_STOP); // consumed in the same 10ms cycle as the first fault
    step(r,30);
    assert(r.Holding_Jaw.Get_Grap_Move_Status()==Enum_Grap_Move_Fault);
    r.Holding_Jaw.XY_Motion.Stop_Confirmed=true;
    step(r,100);
    assert(!r.Sorting_Handshake.Get_Stop_Ack_Pending());
    assert(sent_count(Sorting_Vision_Mode_MCU_STOP_ACK)==1U);
    assert(!r.Request_Sorting_Start());
    // 当前软件原点与待机点均为 (0,0)。
    assert(r.Confirm_XY_Origin_mm(0,0)==HAL_OK);step(r,1000);
    assert(r.Request_Sorting_Start());step(r,30);
    assert(r.Get_Sorting_Running());
    std::puts("PASS: faulted but stopped axes can synchronize STOP while START requires explicit origin recovery");
}

static void test_storage_completion_and_tx_busy()
{
    auto r=ready_robot();queue(Sorting_Vision_Mode_STORAGE_TEST,0,0,2);
    step(r,100);
    assert(sent_count(Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE)==0U);
    tx_status=HAL_BUSY;step(r,1800,false,true);
    assert(r.Sorting_Storage.Get_Disable_Ready());
    assert(sent_count(Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE)==0U);
    tx_status=HAL_OK;step(r,30);
    assert(sent_count(Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE)==1U);
    step(r,100);assert(sent_count(Sorting_Vision_Mode_MCU_STORAGE_TEST_DONE)==1U);
    std::puts("PASS: robot stores mechanical completion across TX busy and publishes storage done once");
}

// 经真实 Robot/握手/抓取状态机读取实际下发的 Yaw，避免只测试一份重复公式。
static float run_single_yaw_pick(float axis_deg)
{
    auto r=ready_robot();
    assert(r.Request_Sorting_Start());
    step(r,30);
    const size_t yaw_before=r.Holding_Jaw.Yaw_Servo_Motor.Commands.size();
    queue(Sorting_Vision_Mode_PICK_PREOPEN,20);
    queue(Sorting_Vision_Mode_PICK_TARGET,-10,10,0,axis_deg);
    for(unsigned i=0;i<2500 && r.Holding_Jaw.Yaw_Servo_Motor.Commands.size()==yaw_before;++i)
        step(r,1,true);
    assert(sent_count(Sorting_Vision_Mode_MCU_PICK_PREOPEN_ACK)==1U);
    assert(sent_count(Sorting_Vision_Mode_MCU_PICK_TARGET_ACK)==1U);
    assert(sent_count(Sorting_Vision_Mode_MCU_ERROR)==0U);
    assert(r.Holding_Jaw.Yaw_Servo_Motor.Commands.size()==yaw_before+1U);
    // 已删除分支事务，整个抓取不应出现旧的 0x91 分支应答。
    assert(sent_count(static_cast<Enum_Sorting_Vision_Mode>(0x91))==0U);
    return r.Holding_Jaw.Yaw_Servo_Motor.Commands.back().Value*180.0f/PI;
}

static void test_single_yaw_full_cycle()
{
    for(int axis=-90;axis<90;++axis)
    {
        const float command=run_single_yaw_pick(static_cast<float>(axis));
        assert(std::isfinite(command));
        assert(command>=-135.0f && command<=135.0f);
    }
    assert(std::fabs(run_single_yaw_pick(-82.0f)-(-19.156234f))<0.001f);
    for(float axis: {-82.0f,-30.0f,0.0f,26.0f,60.0f,89.0f})
    {
        const float command=run_single_yaw_pick(axis);
        assert(std::fabs(run_single_yaw_pick(axis+180.0f)-command)<0.001f);
        assert(std::fabs(run_single_yaw_pick(axis-180.0f)-command)<0.001f);
    }
    std::puts("PASS: single-yaw picks cover 180 degrees, preserve periodicity and map -82 to -19.156 degrees without a branch transaction");
}

int main(int argc,char **argv)
{
    test_single_yaw_full_cycle();
    if(argc==2 && std::strcmp(argv[1],"--single-yaw-only")==0) return 0;
    test_calibration_start_returns_home();test_fault_stopped_ack_and_origin_recovery();test_storage_completion_and_tx_busy();
    std::puts("All robot scheduling regressions passed.");
}
