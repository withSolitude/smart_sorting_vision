#include "3_Robot/holding_jaw.h"
#include "3_Robot/edge_motion_config.h"
#include "3_Robot/sorting_storage.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

FDCAN_HandleTypeDef hfdcan1{1}, hfdcan2{2}, hfdcan3{3};
UART_HandleTypeDef huart3, huart4;
static uint32_t tick=0;
static constexpr float turn_x=Edge_Motion_Config::Turn_Boundary_X_mm -
    static_cast<float>(XY_Config::Position_Tolerance_mm) - Edge_Motion_Config::Turn_Extra_Clearance_mm;
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
    assert(j.XY_Motion.Moves[0].X==turn_x && j.XY_Motion.Moves[0].Y==20.0f);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before); // 退到边界前不转Yaw
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_X_Y_Start);
    assert(j.Claw_Servo_Motor.Commands.size()==claw_before);
    // 仅“已完成”标志不够：反馈坐标仍在边缘内时不能开始转向。
    j.XY_Motion.Motion_Finished=true;
    jaw_step(j,100);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before);
    j.XY_Motion.Arrive();
    jaw_step(j,100);
    assert(j.XY_Motion.Moves.size()==2U);
    assert(j.XY_Motion.Moves[1].X==turn_x && j.XY_Motion.Moves[1].Y==10.0f);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before+1U);
    const uint32_t yaw_tick=j.Yaw_Servo_Motor.Commands.back().Tick;
    j.XY_Motion.Arrive();
    jaw_step(j,400);
    assert(j.XY_Motion.Moves.size()==2U); // XY先到，也必须等Yaw完成
    jaw_step(j,100);
    assert(j.XY_Motion.Moves.size()==3U);
    assert(j.XY_Motion.Moves[2].X==-10.0f && j.XY_Motion.Moves[2].Y==10.0f);
    assert(j.XY_Motion.Moves[2].Tick-yaw_tick>=500U);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before+1U); // 进边缘时只移X
    j.XY_Motion.Arrive();
    jaw_step(j,600);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_Pre_Clip);
    std::puts("PASS: edge route waits for measured retreat, turns outside, then enters with settled yaw");
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
    jaw_step(j,4000,true);
    assert(j.Get_Disable_Ready());
    assert(std::fabs(j.XY_Motion.X)<0.001f && std::fabs(j.XY_Motion.Y-20)<0.001f);
    std::puts("PASS: Disable label cannot bypass actual standby XY, lift clearance or servo settling");
}

static void test_jaw_outside_target_and_boundary()
{
    for(float target_x: {-40.0f,-20.0f,-20.1f})
    {
        auto j=ready_jaw();
        const size_t yaw_before=j.Yaw_Servo_Motor.Commands.size();
        assert(j.Start_Grap(target_x,10,60.0f*PI/180.0f,20,true)==HAL_OK);
        jaw_step(j,200);
        assert(j.XY_Motion.Moves.size()==1U);
        assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before);
        j.XY_Motion.Arrive();
        jaw_step(j,100);
        assert(j.XY_Motion.Moves.size()==2U);
        assert(j.XY_Motion.Moves.back().X==(target_x>turn_x?turn_x:target_x));
        assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before+1U);
        j.XY_Motion.Arrive();
        jaw_step(j,500);
        if(target_x>turn_x)
        {
            // 命令目标x=-20也先在内侧完成Yaw，最后只移X走完容差余量。
            assert(j.XY_Motion.Moves.size()==3U);
            j.XY_Motion.Arrive();
            jaw_step(j,600);
        }
        else jaw_step(j,500);
        assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_Pre_Clip);
        assert(j.XY_Motion.Moves.size()==(target_x>turn_x?3U:2U));
    }
    std::puts("PASS: targets at or near x=-20 prepare yaw inside the measured-position safety margin");
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
    jaw_step(j,5000,true);
    assert(j.Get_Disable_Ready());
    std::puts("PASS: yaw calibration and retried stop preserve standby override until explicit return completes");
}

static void test_jaw_edge_uart_busy()
{
    auto j=ready_jaw();
    const size_t yaw_before=j.Yaw_Servo_Motor.Commands.size();
    assert(j.Start_Grap(-10,10,60.0f*PI/180.0f,20,true)==HAL_OK);
    jaw_step(j,200);
    assert(j.XY_Motion.Moves.size()==1U);
    j.XY_Motion.Arrive();
    huart3.gState=HAL_UART_STATE_BUSY_TX;
    jaw_step(j,1000,true);
    assert(j.XY_Motion.Moves.size()==2U); // UART忙时停在边界，不能提前移入目标X
    assert(j.XY_Motion.X==turn_x);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before);
    huart3.gState=HAL_UART_STATE_READY;
    jaw_step(j,100);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before+1U);
    const uint32_t yaw_tick=j.Yaw_Servo_Motor.Commands.back().Tick;
    jaw_step(j,400,true);
    assert(j.XY_Motion.Moves.size()==2U);
    jaw_step(j,100,true);
    assert(j.XY_Motion.Moves.size()==3U);
    assert(j.XY_Motion.Moves.back().Tick-yaw_tick>=500U);
    std::puts("PASS: UART BUSY cannot advance edge entry or start yaw settling before submission");
}

static void test_jaw_busy_position_changes_side()
{
    auto j=ready_jaw();
    const size_t yaw_before=j.Yaw_Servo_Motor.Commands.size();
    assert(j.Start_Grap(-10,10,60.0f*PI/180.0f,20,true)==HAL_OK);
    // 模拟上一条XY命令仍在执行：最初反馈在安全侧，准备联动时驱动器返回BUSY。
    j.XY_Motion.X=-30.0f;
    j.XY_Motion.Move_Status=HAL_BUSY;
    jaw_step(j,200);
    assert(j.XY_Motion.Moves.empty());
    // BUSY结束时旧命令已经移到边缘内，必须重新退边界，不能照旧步骤直接转Yaw。
    j.XY_Motion.X=0.0f;
    j.XY_Motion.Move_Status=HAL_OK;
    jaw_step(j,200);
    assert(j.XY_Motion.Moves.size()==1U);
    assert(j.XY_Motion.Moves.back().X==turn_x && j.XY_Motion.Moves.back().Y==20.0f);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before);
    // 实测-19.8仍在危险带：即使错误地把Finished置真也必须禁止转Yaw。
    j.XY_Motion.Arrive();
    j.XY_Motion.X=-19.8f;
    jaw_step(j,100);
    assert(j.XY_Motion.Moves.size()==1U);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before);
    // 指令是-20.31，因此到位误差向外0.2mm后仍在安全侧，正常推进不死等。
    j.XY_Motion.X=turn_x+0.2f;
    jaw_step(j,100);
    assert(j.XY_Motion.Moves.size()==2U);
    assert(j.Yaw_Servo_Motor.Commands.size()==yaw_before+1U);
    std::puts("PASS: BUSY retries recheck X; yaw needs measured X<=-20 despite coordinate tolerance");
}

static void test_jaw_full_cycle_routes_reset()
{
    auto j=ready_jaw();
    j.TOF050F.Distance=3;
    assert(j.Start_Grap(-10,10,60.0f*PI/180.0f,20,true)==HAL_OK);
    jaw_step(j,20000,true);
    assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_Finish);
    assert(j.XY_Motion.Moves.size()==8U);
    assert(j.XY_Motion.Moves[3].X==turn_x && j.XY_Motion.Moves[3].Y==10.0f);
    assert(j.XY_Motion.Moves[4].X==turn_x && j.XY_Motion.Moves[4].Y==4.0f);
    assert(j.XY_Motion.Moves[5].X==0.0f && j.XY_Motion.Moves[5].Y==4.0f);
    assert(j.XY_Motion.Moves[6].X==turn_x && j.XY_Motion.Moves[6].Y==4.0f);
    assert(j.XY_Motion.Moves[7].X==-72.0f && j.XY_Motion.Moves[7].Y==0.0f);
    // 固定托盘和识别姿态的实际舵机角不受上位机抓取40~80°规则强夹。
    assert(j.Yaw_Servo_Motor.Commands.size()==4U);
    assert(std::fabs(j.Yaw_Servo_Motor.Commands[2].Value-90.0f*PI/180.0f)<0.0001f);
    assert(std::fabs(j.Yaw_Servo_Motor.Commands[3].Value-110.0f*PI/180.0f)<0.0001f);
    assert(j.Request_Return_To_Disable()==HAL_OK);
    jaw_step(j,5000,true);
    assert(j.Get_Disable_Ready());
    assert(j.XY_Motion.Moves.size()==10U);
    assert(j.Start_Grap(-40,10,60.0f*PI/180.0f,20,true)==HAL_OK);
    jaw_step(j,200);
    assert(j.XY_Motion.Moves.size()==11U);
    assert(j.XY_Motion.Moves.back().X==turn_x && j.XY_Motion.Moves.back().Y==20.0f);
    std::puts("PASS: pick, tray, recognition, return and next pick each start their own waypoint sequence");
}

static void test_jaw_retreat_y_feedback_margin()
{
    for(float observed_y: {XY_Config::Y_Min_mm-0.01f,XY_Config::Y_Max_mm+0.01f})
    {
        auto j=ready_jaw();
        assert(j.Start_Grap(-10,10,60.0f*PI/180.0f,20,true)==HAL_OK);
        j.XY_Motion.Y=observed_y;
        jaw_step(j,200);
        assert(j.Get_Grap_Move_Status()==Enum_Grap_Move_X_Y_Start);
        assert(j.XY_Motion.Moves.size()==1U);
        const auto move=j.XY_Motion.Moves.back();
        assert(move.X==turn_x);
        assert(move.Y>=XY_Config::Y_Min_mm && move.Y<=XY_Config::Y_Max_mm);
        assert(std::fabs(move.Y-observed_y)<0.011f);
    }
    std::puts("PASS: retreat clamps small measured Y boundary error into the legal command workspace");
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
    test_jaw_async_motion();test_jaw_true_disable_and_settle();test_jaw_outside_target_and_boundary();test_jaw_fault_stop_retry();test_jaw_busy_deadline();test_jaw_yaw_calibration_override();test_jaw_edge_uart_busy();test_jaw_busy_position_changes_side();test_jaw_full_cycle_routes_reset();test_jaw_retreat_y_feedback_margin();
    test_storage_delivery_and_done();test_storage_feedback_freshness();test_storage_stop_and_uart_deadline();test_storage_can_type_filter();test_storage_reversed_output_direction();
    test_storage_fault_cannot_auto_enable();
    std::puts("All upper-layer integration regressions passed.");
}
