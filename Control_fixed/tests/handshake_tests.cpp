#include "4_Interaction/ita_sorting_handshake.h"
#include "test_hardware.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

FDCAN_HandleTypeDef hfdcan1{1}, hfdcan2{2}, hfdcan3{3};
UART_HandleTypeDef huart3,huart4;
uint32_t HAL_GetTick() { return 0; }
void HAL_Delay(uint32_t) {}
static std::deque<Struct_Sorting_Vision_UART_Data> incoming;
struct Sent { Enum_Sorting_Vision_Mode Mode; float X,Y; uint8_t Box; };
static std::vector<Sent> sent;
static HAL_StatusTypeDef tx_status=HAL_OK;
bool Class_Sorting_Vision_UART::Pop_Command(Struct_Sorting_Vision_UART_Data &data)
{
    if(incoming.empty()) return false;
    data=incoming.front();incoming.pop_front();return true;
}
void Class_Sorting_Vision_UART::Clear_Command_Queue() { incoming.clear(); }
HAL_StatusTypeDef Class_Sorting_Vision_UART::Send_Frame(const Enum_Sorting_Vision_Mode &mode,
    const float &x,const float &y,const uint8_t &box)
{
    if(tx_status==HAL_OK) sent.push_back({mode,x,y,box});
    return tx_status;
}
int8_t Class_Sorting_Vision_UART::Yaw_Deg_To_Raw(const float &angle)
{ return static_cast<int8_t>(std::round(angle)); }

struct Rig
{
    Class_Sorting_Vision_UART comm;
    Class_Sorting_Handshake handshake;
    Rig() { incoming.clear();sent.clear();tx_status=HAL_OK;handshake.Init(&comm); }
    void queue(Enum_Sorting_Vision_Mode mode,float x=0,float y=0,uint8_t box=0)
    {
        Struct_Sorting_Vision_UART_Data d;d.Mode=mode;d.X_mm=x;d.Y_mm=y;d.Box_ID=box;
        d.Yaw_Branch=box;incoming.push_back(d);
    }
    void command(Enum_Sorting_Vision_Mode mode,float x=0,float y=0,uint8_t box=0)
    { queue(mode,x,y,box);handshake.Process(); }
};

static void test_storage_exclusion_and_done_retry()
{
    Rig r;r.command(Sorting_Vision_Mode_STORAGE_TEST,0,0,2);
    uint8_t box=0;
    assert(r.handshake.Pop_Storage_Test_Request(box) && box==2);
    assert(!r.handshake.Request_Start_Work());
    r.command(Sorting_Vision_Mode_CALIB_MOVE,-5,5);
    assert(sent.back().Mode==Sorting_Vision_Mode_MCU_BUSY);
    Struct_Sorting_Calib_Request xy;
    assert(!r.handshake.Pop_Calib_Request(xy));
    r.command(Sorting_Vision_Mode_CALIB_YAW,30,0);
    assert(sent.back().Mode==Sorting_Vision_Mode_MCU_BUSY);
    Struct_Sorting_Yaw_Calib_Request yaw;
    assert(!r.handshake.Pop_Yaw_Calib_Request(yaw));
    assert(!r.handshake.Notify_Storage_Test_Done(3)); // wrong task cannot clear the lock
    tx_status=HAL_BUSY;
    assert(!r.handshake.Notify_Storage_Test_Done(2));
    tx_status=HAL_OK;
    r.command(Sorting_Vision_Mode_CALIB_MOVE,-5,5);
    assert(sent.back().Mode==Sorting_Vision_Mode_MCU_BUSY);
    assert(r.handshake.Notify_Storage_Test_Done(2));
    assert(!r.handshake.Notify_Storage_Test_Done(2));
    r.command(Sorting_Vision_Mode_CALIB_MOVE,-5,5);
    assert(r.handshake.Get_State()==Sorting_Handshake_State_CALIB_EXECUTING);
    std::puts("PASS: storage test excludes calibration/start and retains task ownership across TX busy");
}

static void test_stop_cancels_and_defers_ack()
{
    Rig r;r.command(Sorting_Vision_Mode_STORAGE_TEST,0,0,2);
    r.queue(Sorting_Vision_Mode_STOP);
    r.queue(Sorting_Vision_Mode_CALIB_MOVE,-5,5); // same batch is invalid after STOP
    r.handshake.Process();
    uint8_t box=0;Struct_Sorting_Calib_Request xy;
    assert(!r.handshake.Pop_Storage_Test_Request(box));
    assert(!r.handshake.Pop_Calib_Request(xy));
    assert(!r.handshake.Notify_Storage_Test_Done(2));
    assert(r.handshake.Get_Stop_Ack_Pending());
    assert(r.handshake.Pop_Stop_Request());
    assert(!r.handshake.Pop_Stop_Request());
    r.command(Sorting_Vision_Mode_CALIB_YAW,20,0);
    assert(sent.back().Mode==Sorting_Vision_Mode_MCU_BUSY);
    tx_status=HAL_BUSY;
    assert(!r.handshake.Notify_Stop_Ready());
    assert(r.handshake.Get_Stop_Ack_Pending());
    tx_status=HAL_OK;
    assert(r.handshake.Notify_Stop_Ready());
    assert(!r.handshake.Get_Stop_Ack_Pending());
    r.command(Sorting_Vision_Mode_STORAGE_TEST,0,0,4);
    assert(r.handshake.Pop_Storage_Test_Request(box) && box==4);
    std::puts("PASS: STOP cancels independent tasks and queued commands, ACK waits for successful transmission");
}

static void test_completion_notifications_preserve_state()
{
    Rig r;assert(r.handshake.Request_Start_Work());
    r.command(Sorting_Vision_Mode_PICK_PREOPEN,20);
    r.command(Sorting_Vision_Mode_PICK_TARGET,-10,10);
    Struct_Sorting_Pick_Request pick;
    assert(r.handshake.Pop_Pick_Request(pick) && pick.Preopen_Valid);
    tx_status=HAL_BUSY;
    assert(!r.handshake.Notify_Recognition_Ready());
    assert(r.handshake.Get_State()==Sorting_Handshake_State_PICK_EXECUTING);
    tx_status=HAL_OK;assert(r.handshake.Notify_Recognition_Ready());
    r.command(Sorting_Vision_Mode_SORT_RESULT,0,0,5);
    Struct_Sorting_Result_Request result;
    assert(r.handshake.Pop_Sort_Request(result) && result.Box_ID==5);
    tx_status=HAL_BUSY;assert(!r.handshake.Notify_Sort_Done());
    assert(r.handshake.Get_State()==Sorting_Handshake_State_SORT_EXECUTING);
    tx_status=HAL_OK;assert(r.handshake.Notify_Sort_Done());
    r.command(Sorting_Vision_Mode_SORT_RESULT,0,0,5);
    assert(sent.back().Mode==Sorting_Vision_Mode_MCU_SORT_DONE);
    assert(!r.handshake.Pop_Sort_Request(result));
    r.command(Sorting_Vision_Mode_STOP);assert(r.handshake.Notify_Stop_Ready());
    r.command(Sorting_Vision_Mode_CALIB_MOVE,-5,5);
    tx_status=HAL_BUSY;assert(!r.handshake.Notify_Calib_Reached(-5,5));
    assert(r.handshake.Get_State()==Sorting_Handshake_State_CALIB_EXECUTING);
    tx_status=HAL_OK;assert(r.handshake.Notify_Calib_Reached(-5,5));
    r.command(Sorting_Vision_Mode_CALIB_YAW,30,0);
    tx_status=HAL_BUSY;assert(!r.handshake.Notify_Yaw_Calib_Reached(30));
    assert(r.handshake.Get_State()==Sorting_Handshake_State_CALIB_YAW_EXECUTING);
    tx_status=HAL_OK;assert(r.handshake.Notify_Yaw_Calib_Reached(30));
    std::puts("PASS: recognition/sort completion retains state during TX busy and duplicates never rerun mechanics");
}

static void test_stop_clears_retry_and_preopen()
{
    Rig r;assert(r.handshake.Request_Start_Work());
    r.command(Sorting_Vision_Mode_PICK_PREOPEN,20);
    r.command(Sorting_Vision_Mode_PICK_YAW_BRANCH,0,0,1);
    r.command(Sorting_Vision_Mode_PICK_TARGET,-10,10);
    assert(r.handshake.Notify_Recognition_Ready());
    r.command(Sorting_Vision_Mode_RETRY_PICK);
    assert(r.handshake.Get_State()==Sorting_Handshake_State_RETRY_RETURNING);
    r.command(Sorting_Vision_Mode_STOP);
    assert(!r.handshake.Pop_Retry_Pick_Request());
    assert(r.handshake.Notify_Stop_Ready());
    assert(r.handshake.Request_Start_Work());
    r.command(Sorting_Vision_Mode_PICK_TARGET,-5,5);
    Struct_Sorting_Pick_Request pick;
    assert(r.handshake.Pop_Pick_Request(pick));
    assert(!pick.Preopen_Valid && pick.Yaw_Branch==0);
    std::puts("PASS: STOP clears retry, per-item preopen and yaw branch before the next transaction");
}

static void test_simultaneous_fault_preserves_explicit_stop()
{
    Rig r;r.command(Sorting_Vision_Mode_STOP);
    r.handshake.Notify_Error(Sorting_Vision_Error_Grap_XY_Position_Invalid);
    assert(r.handshake.Get_State()==Sorting_Handshake_State_STOPPED);
    assert(r.handshake.Get_Stop_Ack_Pending());
    bool explicit_recovery=false;
    assert(r.handshake.Pop_Stop_Request(&explicit_recovery) && explicit_recovery);
    assert(r.handshake.Notify_Stop_Ready());
    r.handshake.Notify_Error(Sorting_Vision_Error_Storage_Runtime);
    explicit_recovery=true;
    assert(r.handshake.Pop_Stop_Request(&explicit_recovery) && !explicit_recovery);
    std::puts("PASS: same-cycle fault preserves operator STOP recovery while automatic faults do not authorize recovery");
}

static void test_error_notification_retries_before_start()
{
    Rig r;tx_status=HAL_BUSY;
    r.handshake.Notify_Error(Sorting_Vision_Error_Storage_Runtime);
    r.handshake.Process();assert(sent.empty());
    r.command(Sorting_Vision_Mode_STOP);
    tx_status=HAL_OK;
    assert(r.handshake.Notify_Stop_Ready());
    assert(!r.handshake.Request_Start_Work()); // ERROR still unqueued despite STOP ACK
    r.handshake.Process();
    assert(sent.back().Mode==Sorting_Vision_Mode_MCU_ERROR && sent.back().Box==Sorting_Vision_Error_Storage_Runtime);
    const size_t sent_before=sent.size();r.handshake.Process();assert(sent.size()==sent_before);
    assert(r.handshake.Request_Start_Work());
    std::puts("PASS: error notification survives TX busy and blocks START until queued exactly once");
}

int main()
{
    test_storage_exclusion_and_done_retry();test_stop_cancels_and_defers_ack();
    test_completion_notifications_preserve_state();test_stop_clears_retry_and_preopen();
    test_simultaneous_fault_preserves_explicit_stop();test_error_notification_retries_before_start();
    std::puts("All handshake regressions passed.");
}
