#include "3_Robot/holding_jaw.h"
#include "3_Robot/sorting_storage.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

FDCAN_HandleTypeDef hfdcan1{1}, hfdcan2{2}, hfdcan3{3};
UART_HandleTypeDef huart3, huart4;
static uint32_t tick=0;
uint32_t HAL_GetTick() { return tick; }
void HAL_Delay(uint32_t ms) { tick+=ms; }

static void jaw_step(Class_Holding_Jaw &j, uint32_t ms, bool arrive=false)
{
    for(uint32_t i=0;i<ms;++i)
    {
        ++tick;
        if(arrive) j.XY_Motion.Arrive();
        if(tick%20U==0U) ++j.TOF050F.Data.Rx_Count;
        j.TIM_1ms_PeriodElapsedCallback();
        if(tick%100U==0U) j.TIM_100ms_PeriodElapsedCallback();
    }
}

static Class_Holding_Jaw ready_jaw()
{
    tick=0; huart3.gState=HAL_UART_STATE_READY;
    Class_Holding_Jaw j;
    j.Init();
    assert(!j.Get_Disable_Ready());
    jaw_step(j,1000);
    assert(j.Get_Disable_Ready());
    return j;
}

static void test_jaw_async_motion()
{
    auto j=ready_jaw();
    const size_t yaw_before=j.Yaw_Servo_Motor.Commands.size();
    const size_t claw_before=j.Claw_Servo_Motor.Commands.size();
    assert(j.Start_Grap(-10,10,0,20,true)==HAL_OK);
    j.XY_Motion.Move_Status=HAL_BUSY;
    jaw_step(j,1000);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_X_Y_Start);
    assert(j.XY_Motion.Moves.empty());
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before);
    assert(j.Claw_Servo_Motor.Commands.size()==claw_before);
    j.XY_Motion.Move_Status=HAL_OK;
    jaw_step(j,1500);
    assert(j.XY_Motion.Moves.size()==1U);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_X_Y_Start);
    assert(j.Claw_Servo_Motor.Commands.size()==claw_before);
    j.XY_Motion.Arrive();
    jaw_step(j,100);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_Pre_Clip);
    std::puts("PASS: jaw retries asynchronous XY busy and cannot preclose before measured arrival");
}

static void test_jaw_true_disable_and_settle()
{
    tick=0;huart3.gState=HAL_UART_STATE_READY;
    Class_Holding_Jaw j;j.Init();
    j.XY_Motion.X=-10;j.XY_Motion.Y=10;
    jaw_step(j,1000);
    assert(!j.Get_Disable_Ready());
    assert(j.Start_Grap(-5,5,0,20,true)==HAL_BUSY);
    assert(j.Request_Return_To_Disable()==HAL_OK);
    const size_t claw_before=j.Claw_Servo_Motor.Commands.size();
    jaw_step(j,400);
    assert(j.XY_Motion.Moves.empty()); // lift must finish before translating
    jaw_step(j,500);
    assert(j.XY_Motion.Moves.size()==1U);
    assert(j.Claw_Servo_Motor.Commands.size()==claw_before);
    assert(!j.Get_Disable_Ready());
    jaw_step(j,500);
    assert(j.Claw_Servo_Motor.Commands.size()==claw_before);
    j.XY_Motion.Arrive();
    jaw_step(j,2000);
    assert(j.Get_Disable_Ready());
    // 当前待机点与上电软件原点一致，回待机后应回到 (0,0)。
    assert(std::fabs(j.XY_Motion.X)<0.001f && std::fabs(j.XY_Motion.Y)<0.001f);
    std::puts("PASS: Disable label cannot bypass actual standby XY, lift clearance or servo settling");
}

static void test_jaw_fault_stop_retry()
{
    auto j=ready_jaw();
    assert(j.Start_Grap(-10,10,0,20,true)==HAL_OK);
    jaw_step(j,200);
    const size_t claw_before=j.Claw_Servo_Motor.Commands.size();
    huart3.gState=HAL_UART_STATE_BUSY_TX;
    j.XY_Motion.Fault=true;j.XY_Motion.Position_Valid=false;
    jaw_step(j,10);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_Fault);
    assert(j.XY_Motion.Stop_Calls!=0U);
    assert(j.Claw_Servo_Motor.Commands.size()==claw_before);
    assert(!j.Get_Stop_Ready());
    huart3.gState=HAL_UART_STATE_READY;
    jaw_step(j,10);
    assert(j.Lift_Servo_Motor.Stops_Accepted>0U && j.Yaw_Servo_Motor.Stops_Accepted>0U);
    assert(!j.Get_Stop_Ready());
    j.XY_Motion.Stop_Confirmed=true;
    jaw_step(j,30);
    assert(j.Get_Stop_Ready());
    assert(!j.Get_Disable_Ready());
    std::puts("PASS: XY fault holds claw and retries UART stops; stopped fault differs from restart ready");
}

static void test_jaw_busy_deadline()
{
    auto j=ready_jaw();
    assert(j.Start_Grap(-10,10,0,20,true)==HAL_OK);
    j.XY_Motion.Move_Status=HAL_BUSY;
    jaw_step(j,30100);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_Fault && j.Get_Grap_Failed());
    assert(j.XY_Motion.Moves.empty());
    std::puts("PASS: jaw output submission cannot wait forever on a busy CAN path");
}

static void test_jaw_yaw_calibration_override()
{
    auto j=ready_jaw();
    assert(j.Control_Calibration_Yaw(std::numeric_limits<float>::quiet_NaN(),300)==HAL_ERROR);
    assert(j.Control_Calibration_Yaw(0,0)==HAL_ERROR);
    assert(j.Control_Calibration_Yaw(0,300)==HAL_OK);
    assert(!j.Get_Disable_Ready() && !j.Get_Stop_Ready());
    huart3.gState=HAL_UART_STATE_BUSY_TX;
    assert(j.Stop_Calibration_Yaw()==HAL_BUSY);
    assert(!j.Get_Stop_Ready());
    huart3.gState=HAL_UART_STATE_READY;
    assert(j.Stop_Calibration_Yaw()==HAL_OK);
    jaw_step(j,30);
    assert(j.Get_Stop_Ready() && !j.Get_Disable_Ready());
    assert(j.Request_Return_To_Disable()==HAL_OK);
    jaw_step(j,3000,true);
    assert(j.Get_Disable_Ready());
    std::puts("PASS: yaw calibration and retried stop preserve standby override until explicit return completes");
}

static void storage_step(Class_Sorting_Storage &s,uint32_t ms,bool fresh=true,bool follow=false,
                         bool fresh_ordinary=true)
{
    for(uint32_t i=0;i<ms;++i)
    {
        ++tick;
        if(follow) { s.DM4310.Angle=s.Get_DM4310_Target_Angle();s.DM4310.Omega=0; }
        if(tick%10U==0U)
        {
            if(fresh) ++s.DM4310.Position_Count;
            if(fresh_ordinary) { ++s.DM4310.Feedback_Count;s.DM4310.Feedback_Tick=tick; }
        }
        if(tick%2U==0U) s.TIM_2ms_PeriodElapsedCallback();
        if(tick%10U==0U) s.TIM_10ms_PeriodElapsedCallback();
        if(tick%100U==0U) s.TIM_100ms_Alive_PeriodElapsedCallback();
    }
}

static Class_Sorting_Storage ready_storage()
{
    tick=0;huart4.gState=HAL_UART_STATE_READY;
    Class_Sorting_Storage s;s.Init(&hfdcan1,1,0,&huart4,4);
    assert(!s.Get_Disable_Ready());
    storage_step(s,1000);
    assert(s.Get_Disable_Ready());
    return s;
}

static void test_storage_delivery_and_done()
{
    auto s=ready_storage();
    assert(s.Start_Sort(0)==HAL_ERROR);
    assert(s.Start_Sort(7)==HAL_ERROR);
    assert(s.Start_Sort(2)==HAL_OK);
    const size_t commands_before=s.Tray_Flip_Servo_Motor.Commands.size();
    storage_step(s,200);
    assert(s.Get_Status()==Sorting_Storage_Status_DM_MOVE_TO_BOX);
    assert(s.Tray_Flip_Servo_Motor.Commands.size()==commands_before);
    assert(!s.Pop_Sort_Finished());
    storage_step(s,1800,true,true);
    assert(s.Get_Disable_Ready());
    assert(s.Pop_Sort_Finished());
    assert(!s.Pop_Sort_Finished());
    std::puts("PASS: storage waits for measured box alignment and full tray/DM return before one-shot done");
}

static void test_storage_feedback_freshness()
{
    auto s=ready_storage();
    assert(s.Start_Sort(1)==HAL_OK);
    storage_step(s,15,false);
    assert(s.Get_Status()==Sorting_Storage_Status_DM_MOVE_TO_BOX);
    const size_t commands=s.Tray_Flip_Servo_Motor.Commands.size();
    storage_step(s,200,false);
    assert(s.Tray_Flip_Servo_Motor.Commands.size()==commands);
    assert(s.Get_Fault() && !s.Get_Disable_Ready() && !s.Pop_Sort_Finished());
    assert(!s.Clear_Fault());
    s.Request_Disable();storage_step(s,1200,true,true);
    assert(s.Get_Fault() && s.Get_Disable_Ready());
    assert(s.Start_Sort(1)==HAL_ERROR);
    assert(s.Clear_Fault());
    assert(!s.Get_Fault());
    auto p=ready_storage();
    assert(p.Start_Sort(1)==HAL_OK);
    storage_step(p,220,true,false,false); // output angle fresh, speed/status frozen
    assert(p.Get_Fault() && !p.Pop_Sort_Finished());
    std::puts("PASS: stale matching feedback cannot flip tray; recovery requires explicit fault acknowledgement");
}

static void test_storage_stop_and_uart_deadline()
{
    auto s=ready_storage();
    assert(s.Start_Sort(3)==HAL_OK);
    storage_step(s,100);
    s.Request_Disable();
    for(unsigned i=0;i<60U;++i) { s.Request_Disable();storage_step(s,20,true,true); }
    assert(s.Get_Disable_Ready());
    assert(!s.Pop_Sort_Finished());
    assert(s.Start_Sort(2)==HAL_OK);
    huart4.gState=HAL_UART_STATE_BUSY_TX;
    storage_step(s,3500,true,true);
    assert(s.Get_Fault() && !s.Pop_Sort_Finished());
    std::puts("PASS: repeated STOP is idempotent and UART submission has a finite failure deadline");
}

static void test_storage_can_type_filter()
{
    auto s=ready_storage();
    FDCAN_RxHeaderTypeDef h{};h.IdType=FDCAN_EXTENDED_ID;h.Identifier=0x100;h.DataLength=3;
    s.CAN_RxCpltCallback(&hfdcan1,h);
    assert(s.DM4310.Rx_Calls==0);
    h.IdType=FDCAN_STANDARD_ID;h.Identifier=1;h.DataLength=8;
    h.RxFrameType=FDCAN_REMOTE_FRAME;
    s.CAN_RxCpltCallback(&hfdcan1,h);
    assert(s.DM4310.Rx_Calls==0);
    h.RxFrameType=FDCAN_DATA_FRAME;
    s.CAN_RxCpltCallback(&hfdcan1,h);
    assert(s.DM4310.Rx_Calls==1);
    std::puts("PASS: storage CAN receiver rejects Emm extended frames sharing the CAN1 bus");
}

static void test_storage_reversed_output_direction()
{
    auto s=ready_storage();
    s.Set_DM4310_Output_Direction(-1);
    assert(s.Start_Sort(2)==HAL_OK);
    storage_step(s,20);
    assert(s.Get_DM4310_PID_Torque_Command()>0);
    assert(s.DM4310.Torque<0);
    // For direction=-1 the raw output angle is the negative of the requested
    // mechanical angle. Feedback and torque must use the same convention.
    s.DM4310.Angle=-s.Get_DM4310_Target_Angle();
    storage_step(s,80);
    assert(s.Get_Status()!=Sorting_Storage_Status_DM_MOVE_TO_BOX);
    std::puts("PASS: reversed DM output direction transforms measured position and commanded torque consistently");
}

static void test_storage_fault_cannot_auto_enable()
{
    auto s=ready_storage();
    s.DM4310.Alive_Auto_Enable=true;
    s.DM4310.Control_Status=Motor_DM_Control_Status_DISABLE;
    storage_step(s,20);
    assert(s.Get_Fault());
    const uint32_t alive_before=s.DM4310.Alive_Calls;
    const uint32_t clear_before=s.DM4310.Clear_Error_Calls;
    const uint32_t enter_before=s.DM4310.Enter_Calls;
    storage_step(s,500);
    assert(s.DM4310.Alive_Calls==alive_before);
    assert(s.DM4310.Clear_Error_Calls==clear_before && s.DM4310.Enter_Calls==enter_before);
    assert(s.DM4310.Control_Status==Motor_DM_Control_Status_DISABLE);
    assert(!s.Get_Disable_Ready() && !s.Pop_Sort_Finished());
    s.Request_Disable();
    assert(!s.Get_Disable_Ready());
    s.TIM_100ms_Alive_PeriodElapsedCallback(); // only explicit STOP grants this side effect
    assert(s.DM4310.Alive_Calls>alive_before);
    assert(s.DM4310.Clear_Error_Calls>clear_before && s.DM4310.Enter_Calls>enter_before);
    assert(!s.Get_Disable_Ready() && !s.Pop_Sort_Finished());
    storage_step(s,1200,true,true);
    assert(s.Get_Disable_Ready() && s.Get_Fault());
    assert(!s.Pop_Sort_Finished());
    assert(s.Clear_Fault());
    auto q=ready_storage();
    q.DM4310.Control_Status=Motor_DM_Control_Status_DISABLE;
    storage_step(q,20);assert(q.Get_Fault());
    q.Request_Disable();
    storage_step(q,3500); // recovery cannot re-enable hardware, so it must time out
    assert(q.Get_Status()==Sorting_Storage_Status_ERROR);
    const uint32_t alive_after_failed_recovery=q.DM4310.Alive_Calls;
    q.DM4310.Alive_Auto_Enable=true;
    storage_step(q,500);
    assert(q.DM4310.Alive_Calls==alive_after_failed_recovery);
    assert(q.DM4310.Control_Status==Motor_DM_Control_Status_DISABLE);
    std::puts("PASS: latched storage faults prohibit automatic DM clear/enable until explicit STOP authorizes safe recovery");
}

int main()
{
    test_jaw_async_motion();test_jaw_true_disable_and_settle();test_jaw_fault_stop_retry();test_jaw_busy_deadline();test_jaw_yaw_calibration_override();
    test_storage_delivery_and_done();test_storage_feedback_freshness();test_storage_stop_and_uart_deadline();test_storage_can_type_filter();test_storage_reversed_output_direction();
    test_storage_fault_cannot_auto_enable();
    std::puts("All upper-layer integration regressions passed.");
}
