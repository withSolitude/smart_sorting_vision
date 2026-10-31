#include "3_Robot/xy_motion.h"
#include <cmath>
#include <algorithm>

namespace
{
struct IRQ_Guard
{
    uint32_t Saved = __get_PRIMASK();
    IRQ_Guard() { __disable_irq(); }
    ~IRQ_Guard() { __set_PRIMASK(Saved); }
};
bool Fresh(uint32_t now, uint32_t tick)
{
    return static_cast<uint32_t>(now - tick) <= XY_Config::Feedback_Timeout_ms;
}
bool In_Workspace(double x, double y)
{
    return std::isfinite(x) && std::isfinite(y) &&
        x >= XY_Config::X_Min_mm && x <= XY_Config::X_Max_mm &&
        y >= XY_Config::Y_Min_mm && y <= XY_Config::Y_Max_mm;
}
}

void Class_XY_Motion::XY_To_Motor_mm(double x, double y, double &a, double &b)
{
    const double theta = XY_Config::Frame_Rotation_Deg * XY_Config::Pi / 180.0;
    const double c = std::cos(theta), s = std::sin(theta);
    const double u = c*x + s*y, v = -s*x + c*y;
    a = XY_Config::Direction_A * (u + v);
    b = XY_Config::Direction_B * (u - v);
}

void Class_XY_Motion::Motor_To_XY_mm(double a, double b, double &x, double &y)
{
    a *= XY_Config::Direction_A;
    b *= XY_Config::Direction_B;
    const double u = (a+b)*0.5, v = (a-b)*0.5;
    const double theta = XY_Config::Frame_Rotation_Deg * XY_Config::Pi / 180.0;
    const double c = std::cos(theta), s = std::sin(theta);
    x = c*u - s*v;
    y = s*u + c*v;
}

bool Class_XY_Motion::Config_Valid() const
{
    return XY_Config::CAN_Bus() != nullptr &&
        (XY_Config::Address_A == 1U || XY_Config::Address_B == 1U) && XY_Config::Address_A != 0U &&
        XY_Config::Address_B != 0U && XY_Config::Address_A != XY_Config::Address_B &&
        std::isfinite(XY_Config::Travel_Per_Rev_A_mm) && XY_Config::Travel_Per_Rev_A_mm > 0 &&
        std::isfinite(XY_Config::Travel_Per_Rev_B_mm) && XY_Config::Travel_Per_Rev_B_mm > 0 &&
        std::isfinite(XY_Config::Frame_Rotation_Deg) &&
        std::abs(XY_Config::Direction_A) == 1 && std::abs(XY_Config::Direction_B) == 1 &&
        XY_Config::Pulses_Per_Rev > 0U && XY_Config::Max_RPM > 0U &&
        XY_Config::Encoder_Unwrap_Max_Gap_ms * XY_Config::Max_RPM < 30000U;
}

void Class_XY_Motion::Init()
{
    IRQ_Guard guard;
    Motor_A.Init(XY_Config::CAN_Bus(), XY_Config::Address_A, XY_Config::Encoder_Unwrap_Max_Gap_ms);
    Motor_B.Init(XY_Config::CAN_Bus(), XY_Config::Address_B, XY_Config::Encoder_Unwrap_Max_Gap_ms);
    Phase = State::Stop_A; // MCU复位不能让仍在通电的驱动继续执行旧运动
    Phase_Tick = HAL_GetTick();
    Fault = Config_Valid() ? XY_Fault::None : XY_Fault::Configuration;
    Origin_Valid = false;
    Now_X = Now_Y = 0;
    Last_Position_A_Count = Last_Position_B_Count = 0;
    Poll_Tick = HAL_GetTick();
    Poll_Slot = 0;
    Stable = false;
    Track_Enabled = false;
    Motion_Completed = false;
    Stop_Sequence_Confirmed = false;
}

void Class_XY_Motion::CAN_RxCpltCallback(const FDCAN_RxHeaderTypeDef &header, const uint8_t *data)
{
    Motor_A.CAN_RxCpltCallback(header, data);
    Motor_B.CAN_RxCpltCallback(header, data);
}

bool Class_XY_Motion::Feedback_Fresh() const
{
    const auto a = Motor_A.Get_Data(), b = Motor_B.Get_Data();
    const uint32_t now = HAL_GetTick();
    return a.Position_Valid && b.Position_Valid && a.Status_Valid && b.Status_Valid &&
        a.Speed_Valid && b.Speed_Valid && Fresh(now,a.Position_Timestamp_ms) &&
        Fresh(now,b.Position_Timestamp_ms) && Fresh(now,a.Status_Timestamp_ms) &&
        Fresh(now,b.Status_Timestamp_ms) && Fresh(now,a.Speed_Timestamp_ms) &&
        Fresh(now,b.Speed_Timestamp_ms);
}

bool Class_XY_Motion::Get_Position_Valid() const
{
    IRQ_Guard guard;
    return Origin_Valid && !Get_Fault() && Feedback_Fresh();
}

bool Class_XY_Motion::Get_Motion_Finished() const
{
    IRQ_Guard guard;
    return Motion_Completed && Phase == State::Idle && Get_Position_Valid() && Get_Stop_Confirmed();
}

bool Class_XY_Motion::Get_Stop_Confirmed() const
{
    IRQ_Guard guard;
    const auto a = Motor_A.Get_Data(), b = Motor_B.Get_Data();
    return Phase == State::Idle && Feedback_Fresh() &&
        std::abs(a.Speed_RPM) <= XY_Config::Stopped_RPM &&
        std::abs(b.Speed_RPM) <= XY_Config::Stopped_RPM;
}

HAL_StatusTypeDef Class_XY_Motion::Start_Collision_Homing()
{
    IRQ_Guard guard;
    if (!XY_Config::Driver_Collision_Homing_Enabled ||
        XY_Config::Driver_Homing_Mode != 2U || !Config_Valid()) return HAL_ERROR;
    if (Homing_Active || Phase != State::Idle) return HAL_BUSY;
    if (!Feedback_Fresh() || !Get_Stop_Confirmed()) return HAL_BUSY;
    if (Motor_A.Get_Data().Stall || Motor_B.Get_Data().Stall ||
        Motor_A.Get_Data().Stall_Protection || Motor_B.Get_Data().Stall_Protection) return HAL_ERROR;

    Origin_Valid = false;
    Motion_Completed = false;
    Homing_Active = true;
    Stop_Sequence_Confirmed = false;
    Homing_Failed = false;
    Homing_Stop_Requested = false;
    Homing_Tick = HAL_GetTick();
    Homing_Status_Tick = Homing_Tick;
    Set_Phase(State::Home_Enable_A);
    return HAL_OK;
}

void Class_XY_Motion::Stop_Collision_Homing()
{
    IRQ_Guard guard;
    if (!Homing_Active) return;
    Motor_A.Stop_Homing();
    Motor_B.Stop_Homing();
    Homing_Stop_Requested = true;
    Homing_Active = false;
    Homing_Failed = true;
    Origin_Valid = false;
    Fault = XY_Fault::Homing_Failed;
    Begin_Stop();
}

void Class_XY_Motion::Set_Phase(State phase)
{
    Phase = phase;
    Phase_Tick = HAL_GetTick();
}

void Class_XY_Motion::Capture_Feedback_Baselines()
{
    const auto a = Motor_A.Get_Data(), b = Motor_B.Get_Data();
    Position_A_Baseline = a.Position_Count; Position_B_Baseline = b.Position_Count;
    Status_A_Baseline = a.Status_Count; Status_B_Baseline = b.Status_Count;
    Speed_A_Baseline = a.Speed_Count; Speed_B_Baseline = b.Speed_Count;
    Stable = false;
}

HAL_StatusTypeDef Class_XY_Motion::Confirm_Origin_mm(float x, float y)
{
    IRQ_Guard guard;
    if (!XY_Config::Mechanical_Calibration_Confirmed || !Config_Valid() || !In_Workspace(x,y))
        return HAL_ERROR;
    if (!Get_Stop_Confirmed()) return HAL_BUSY;
    const auto a = Motor_A.Get_Data(), b = Motor_B.Get_Data();
    if (a.Stall || b.Stall || a.Stall_Protection || b.Stall_Protection) return HAL_ERROR;
    Origin_A = a.Position_Ticks; Origin_B = b.Position_Ticks;
    Origin_X = Now_X = x; Origin_Y = Now_Y = y;
    Target_A = Origin_A; Target_B = Origin_B;
    Origin_Valid = true;
    Motion_Completed = true;
    Track_Enabled = a.Enabled && b.Enabled;
    Fault = XY_Fault::None;
    Last_A = Origin_A; Last_B = Origin_B;
    Last_Position_A_Count = a.Position_Count; Last_Position_B_Count = b.Position_Count;
    Last_Position_A_Tick = a.Position_Timestamp_ms; Last_Position_B_Tick = b.Position_Timestamp_ms;
    Motor_A.Reset_Encoder_Unwrap(); Motor_B.Reset_Encoder_Unwrap();
    return HAL_OK;
}

HAL_StatusTypeDef Class_XY_Motion::Move_To_mm(float x, float y, float vx, float vy)
{
    IRQ_Guard guard;
    if (!XY_Config::Mechanical_Calibration_Confirmed || !Config_Valid() ||
        !In_Workspace(x,y) || !std::isfinite(vx) || !std::isfinite(vy) || vx <= 0 || vy <= 0 ||
        !Get_Position_Valid()) return HAL_ERROR;
    if (Phase != State::Idle) return HAL_BUSY;
    double a, b;
    XY_To_Motor_mm(x-Origin_X, y-Origin_Y, a, b);
    // 位置指令使用每圈脉冲数，反馈使用每圈 65536 个编码器刻度，二者不能混用。
    const double pa = (Origin_A/65536.0 + a/XY_Config::Travel_Per_Rev_A_mm)*XY_Config::Pulses_Per_Rev;
    const double pb = (Origin_B/65536.0 + b/XY_Config::Travel_Per_Rev_B_mm)*XY_Config::Pulses_Per_Rev;
    if (std::abs(pa) > 4294967295.0 || std::abs(pb) > 4294967295.0 ||
        std::abs(pa*65536.0/XY_Config::Pulses_Per_Rev) > 4294967295.0 ||
        std::abs(pb*65536.0/XY_Config::Pulses_Per_Rev) > 4294967295.0) return HAL_ERROR;
    Target_A = std::llround(pa*65536.0/XY_Config::Pulses_Per_Rev);
    Target_B = std::llround(pb*65536.0/XY_Config::Pulses_Per_Rev);
    const auto ad = Motor_A.Get_Data(), bd = Motor_B.Get_Data();
    Protocol_Error_A_Baseline = ad.Protocol_Error_Count;
    Protocol_Error_B_Baseline = bd.Protocol_Error_Count;
    const double ra = std::abs((Target_A-ad.Position_Ticks)/65536.0);
    const double rb = std::abs((Target_B-bd.Position_Ticks)/65536.0);
    double duration = std::max(std::abs(x-Now_X)/vx, std::abs(y-Now_Y)/vy);
    duration = std::max(duration, std::max(ra,rb)*60.0/XY_Config::Max_RPM);
    duration = std::max(duration, 0.001);
    Speed_A = static_cast<uint16_t>(std::max(1.0, std::min<double>(XY_Config::Max_RPM,std::round(ra*60.0/duration))));
    Speed_B = static_cast<uint16_t>(std::max(1.0, std::min<double>(XY_Config::Max_RPM,std::round(rb*60.0/duration))));
    Move_Tick = HAL_GetTick();
    Motion_Completed = false;
    Stop_Sequence_Confirmed = false;
    Set_Phase(State::Enable_A);
    return HAL_OK; // 仅表示目标已接受；到位必须等两轴真实反馈稳定
}

void Class_XY_Motion::Begin_Stop()
{
    Motion_Completed = false; // 取消完成不等于原目标到位
    Stop_Sequence_Confirmed = false;
    Capture_Feedback_Baselines();
    Set_Phase(State::Stop_A);
}

void Class_XY_Motion::Stop()
{
    IRQ_Guard guard;
    if (Homing_Active)
    {
        Stop_Collision_Homing();
        return;
    }
    // 上电首次仍必须发送双轴STOP，不能仅凭转速零跳过。
    // 已完成STOP且反馈仍新鲜、转速仍为零时，重发命令只保留原安全确认。
    if (Phase == State::Idle && Stop_Sequence_Confirmed && Get_Stop_Confirmed())
    {
        Motion_Completed = false; // 重复STOP也取消上一目标的完成状态，但不重启停车。
        return;
    }
    if (Phase < State::Stop_A) Begin_Stop();
}

void Class_XY_Motion::Fail_Homing()
{
    Motor_A.Stop_Homing();
    Motor_B.Stop_Homing();
    Homing_Active = false;
    Homing_Failed = true;
    Origin_Valid = false;
    Fault = XY_Fault::Homing_Failed;
    Begin_Stop();
}

void Class_XY_Motion::Set_Fault(XY_Fault fault)
{
    if (Homing_Active)
    {
        Fail_Homing();
        return;
    }
    if (Fault == XY_Fault::None) Fault = fault;
    Origin_Valid = false;
    if (Phase < State::Stop_A) Begin_Stop();
}

void Class_XY_Motion::Update_Position()
{
    const auto a = Motor_A.Get_Data(), b = Motor_B.Get_Data();
    if (!a.Position_Valid || !b.Position_Valid) return;
    if (Origin_Valid)
    {
        const auto check_jump = [](int64_t current, int64_t last, uint32_t tick, uint32_t last_tick)
        {
            const double bound = (uint32_t)(tick-last_tick)*XY_Config::Max_RPM*65536.0/60000.0 + 1024.0;
            return std::abs(static_cast<double>(current-last)) > bound;
        };
        if ((a.Position_Count != Last_Position_A_Count && check_jump(a.Position_Ticks,Last_A,a.Position_Timestamp_ms,Last_Position_A_Tick)) ||
            (b.Position_Count != Last_Position_B_Count && check_jump(b.Position_Ticks,Last_B,b.Position_Timestamp_ms,Last_Position_B_Tick)))
        {
            Set_Fault(XY_Fault::Position_Jump);
        }
        double x,y;
        Motor_To_XY_mm((a.Position_Ticks-Origin_A)*XY_Config::Travel_Per_Rev_A_mm/65536.0,
                       (b.Position_Ticks-Origin_B)*XY_Config::Travel_Per_Rev_B_mm/65536.0,x,y);
        Now_X = Origin_X+x; Now_Y = Origin_Y+y;
    }
    Last_A = a.Position_Ticks; Last_B = b.Position_Ticks;
    Last_Position_A_Count = a.Position_Count; Last_Position_B_Count = b.Position_Count;
    Last_Position_A_Tick = a.Position_Timestamp_ms; Last_Position_B_Tick = b.Position_Timestamp_ms;
}

void Class_XY_Motion::Poll()
{
    if ((uint32_t)(HAL_GetTick()-Poll_Tick) < XY_Config::Poll_Interval_ms) return;
    Poll_Tick = HAL_GetTick();
    auto &motor = (Poll_Slot & 1U) ? Motor_B : Motor_A;
    HAL_StatusTypeDef result;
    switch (Poll_Slot/2U)
    {
        case 0: result = motor.Request_Position(); break;
        case 1: result = motor.Request_Status(); break;
        case 2: result = motor.Request_Speed(); break;
        default: result = motor.Request_Encoder(); break;
    }
    if (result == HAL_OK) Poll_Slot = (Poll_Slot+1U)%8U;
}

void Class_XY_Motion::TIM_1ms_PeriodElapsedCallback()
{
    IRQ_Guard guard;
    const uint32_t now = HAL_GetTick();
    Update_Position();
    auto a = Motor_A.Get_Data(), b = Motor_B.Get_Data();
    if (Origin_Valid && !Feedback_Fresh()) Set_Fault(XY_Fault::Feedback_Timeout);
    if (Origin_Valid && (a.Stall || b.Stall || a.Stall_Protection || b.Stall_Protection)) Set_Fault(XY_Fault::Drive_Stall);
    if (Origin_Valid && Track_Enabled && Phase == State::Idle && (!a.Enabled || !b.Enabled))
        Set_Fault(XY_Fault::Drive_Disabled);
    if (Phase >= State::Enable_A && Phase <= State::Moving &&
        (a.Protocol_Error_Count != Protocol_Error_A_Baseline || b.Protocol_Error_Count != Protocol_Error_B_Baseline))
        Set_Fault(XY_Fault::Command_Rejected);
    if (Phase >= State::Enable_A && Phase <= State::Moving &&
        (uint32_t)(now-Move_Tick) >= XY_Config::Move_Timeout_ms) Set_Fault(XY_Fault::Move_Timeout);
    const bool normal_command_phase =
        (Phase >= State::Enable_A && Phase <= State::Wait_Sync) ||
        (Phase >= State::Stop_A && Phase <= State::Stopping);
    if (normal_command_phase &&
        (uint32_t)(now-Phase_Tick) >= XY_Config::Command_Timeout_ms)
    {
        Set_Fault(XY_Fault::Command_Timeout);
        Begin_Stop(); // CAN恢复后继续尝试两轴停止，绝不因超时声称已经停机
    }

    if (Homing_Active && (uint32_t)(now - Homing_Tick) >= XY_Config::Homing_Timeout_ms)
    {
        Fail_Homing();
    }

    if (Homing_Active &&
        (Phase == State::Home_Wait_B || Phase == State::Home_Wait_A) &&
        (uint32_t)(now - Homing_Status_Tick) >= XY_Config::Homing_Status_Poll_ms)
    {
        Homing_Status_Tick = now;
        if (Phase == State::Home_Wait_B) Motor_B.Request_Homing_Status();
        else Motor_A.Request_Homing_Status();
    }

    HAL_StatusTypeDef result = HAL_OK;
    const auto await_ack = [&](const Struct_Motor_EmmV5_Ack &ack, State next)
    {
        if (ack.Count == Ack_Baseline) return;
        if (ack.Status != 0x02U) { Set_Fault(XY_Fault::Command_Rejected); return; }
        Set_Phase(next);
    };
    switch (Phase)
    {
        case State::Idle: break;
        case State::Enable_A:
            Ack_Baseline = a.Enable_Ack.Count; result = Motor_A.Enable(true);
            if (result == HAL_OK) Set_Phase(State::Wait_Enable_A);
            break;
        case State::Wait_Enable_A: await_ack(a.Enable_Ack,State::Enable_B); break;
        case State::Enable_B:
            Ack_Baseline = b.Enable_Ack.Count; result = Motor_B.Enable(true);
            if (result == HAL_OK) Set_Phase(State::Wait_Enable_B);
            break;
        case State::Wait_Enable_B: await_ack(b.Enable_Ack,State::Arm_A); break;
        case State::Arm_A:
            Ack_Baseline = a.Position_Ack.Count;
            result = Motor_A.Send_Position(std::llround(Target_A*XY_Config::Pulses_Per_Rev/65536.0),Speed_A,XY_Config::Acceleration);
            if (result == HAL_OK) Set_Phase(State::Wait_Arm_A);
            break;
        case State::Wait_Arm_A: await_ack(a.Position_Ack,State::Arm_B); break;
        case State::Arm_B:
            Ack_Baseline = b.Position_Ack.Count;
            result = Motor_B.Send_Position(std::llround(Target_B*XY_Config::Pulses_Per_Rev/65536.0),Speed_B,XY_Config::Acceleration);
            if (result == HAL_OK) Set_Phase(State::Wait_Arm_B);
            break;
        case State::Wait_Arm_B: await_ack(b.Position_Ack,State::Sync); break;
        case State::Sync:
            Ack_Baseline = XY_Config::Address_A == 1U ? a.Sync_Ack.Count : b.Sync_Ack.Count;
            result = Class_Motor_EmmV5::Sync(XY_Config::CAN_Bus());
            if (result == HAL_OK) Set_Phase(State::Wait_Sync);
            break;
        case State::Wait_Sync:
            await_ack(XY_Config::Address_A == 1U ? a.Sync_Ack : b.Sync_Ack, State::Moving);
            if (Phase == State::Moving) Capture_Feedback_Baselines();
            break;
        case State::Moving:
        {
            const bool new_feedback = a.Position_Count != Position_A_Baseline && b.Position_Count != Position_B_Baseline &&
                a.Status_Count != Status_A_Baseline && b.Status_Count != Status_B_Baseline &&
                a.Speed_Count != Speed_A_Baseline && b.Speed_Count != Speed_B_Baseline;
            if (new_feedback && (!a.Enabled || !b.Enabled)) { Set_Fault(XY_Fault::Drive_Disabled); break; }
            if (new_feedback && a.Enabled && b.Enabled) Track_Enabled = true;
            const double ea = std::abs((a.Position_Ticks-Target_A)*XY_Config::Travel_Per_Rev_A_mm/65536.0);
            const double eb = std::abs((b.Position_Ticks-Target_B)*XY_Config::Travel_Per_Rev_B_mm/65536.0);
            const bool reached = new_feedback && Feedback_Fresh() && a.Reached && b.Reached &&
                ea <= XY_Config::Position_Tolerance_mm && eb <= XY_Config::Position_Tolerance_mm &&
                std::abs(a.Speed_RPM) <= XY_Config::Stopped_RPM && std::abs(b.Speed_RPM) <= XY_Config::Stopped_RPM;
            if (!reached) Stable = false;
            else if (!Stable)
            {
                Stable = true; Stable_Tick = now;
                Stable_Position_A_Count = a.Position_Count; Stable_Position_B_Count = b.Position_Count;
            }
            else if ((uint32_t)(now-Stable_Tick) >= XY_Config::Settle_ms &&
                     a.Position_Count != Stable_Position_A_Count && b.Position_Count != Stable_Position_B_Count)
            {
                Motion_Completed = true;
                Set_Phase(State::Idle);
            }
            break;
        }
        case State::Stop_A:
            Stop_A_Baseline = a.Stop_Ack.Count; result = Motor_A.Stop();
            if (result == HAL_OK) Set_Phase(State::Stop_B);
            break;
        case State::Stop_B:
            Stop_B_Baseline = b.Stop_Ack.Count; result = Motor_B.Stop();
            if (result == HAL_OK) { Capture_Feedback_Baselines(); Set_Phase(State::Stopping); }
            break;
        case State::Stopping:
        {
            // 电机已失能或处于堵转保护时，FE 可能返回 E2；
            // 只有新反馈同时证明“已失能且转速为零”时，才能接受为停机完成。
            const auto stop_ack_ok = [](const Struct_Motor_EmmV5_Data &d, uint32_t baseline)
            {
                return d.Stop_Ack.Count != baseline && (d.Stop_Ack.Status == 0x02U ||
                    (d.Stop_Ack.Status == 0xE2U && !d.Enabled &&
                     std::abs(d.Speed_RPM) <= XY_Config::Stopped_RPM));
            };
            const bool ack = stop_ack_ok(a,Stop_A_Baseline) && stop_ack_ok(b,Stop_B_Baseline);
            const bool stopped = ack && Feedback_Fresh() && a.Position_Count != Position_A_Baseline &&
                b.Position_Count != Position_B_Baseline && a.Speed_Count != Speed_A_Baseline && b.Speed_Count != Speed_B_Baseline &&
                a.Status_Count != Status_A_Baseline && b.Status_Count != Status_B_Baseline &&
                std::abs(a.Speed_RPM) <= XY_Config::Stopped_RPM && std::abs(b.Speed_RPM) <= XY_Config::Stopped_RPM;
            if (!stopped) Stable = false;
            else if (!Stable)
            {
                Stable = true; Stable_Tick = now;
                Stable_Position_A_Count = a.Position_Count; Stable_Position_B_Count = b.Position_Count;
            }
            else if ((uint32_t)(now-Stable_Tick) >= XY_Config::Settle_ms &&
                     a.Position_Count != Stable_Position_A_Count && b.Position_Count != Stable_Position_B_Count)
            {
                Stop_Sequence_Confirmed = true;
                Set_Phase(State::Idle);
            }
            break;
        }
        case State::Home_Enable_A:
            Ack_Baseline = a.Enable_Ack.Count; result = Motor_A.Enable(true);
            if (result == HAL_OK) Set_Phase(State::Home_Wait_Enable_A);
            break;
        case State::Home_Wait_Enable_A:
            if (a.Enable_Ack.Count != Ack_Baseline)
            {
                if (a.Enable_Ack.Status != 0x02U) Fail_Homing();
                else Set_Phase(State::Home_Enable_B);
            }
            break;
        case State::Home_Enable_B:
            Ack_Baseline = b.Enable_Ack.Count; result = Motor_B.Enable(true);
            if (result == HAL_OK) Set_Phase(State::Home_Wait_Enable_B);
            break;
        case State::Home_Wait_Enable_B:
            if (b.Enable_Ack.Count != Ack_Baseline)
            {
                if (b.Enable_Ack.Status != 0x02U) Fail_Homing();
                else Set_Phase(State::Home_B);
            }
            break;
        case State::Home_B:
            Homing_Ack_Baseline = b.Homing_Ack.Count;
            Homing_Status_B_Baseline = b.Homing_Status_Count;
            result = Motor_B.Trigger_Homing(XY_Config::Driver_Homing_Mode, false);
            if (result == HAL_OK) Set_Phase(State::Home_Wait_B);
            break;
        case State::Home_Wait_B:
            if (b.Homing_Ack.Count != Homing_Ack_Baseline && b.Homing_Ack.Status != 0x02U)
            {
                Fail_Homing();
                break;
            }
            if (b.Homing_Status_Count != Homing_Status_B_Baseline)
            {
                Homing_Status_B_Baseline = b.Homing_Status_Count;
                if (b.Homing_Failed ||
                    (!b.Homing_In_Progress && b.Position_Valid && b.Speed_Valid &&
                     std::abs(b.Speed_RPM) <= XY_Config::Stopped_RPM))
                {
                    if (b.Homing_Failed) { Fail_Homing(); break; }
                    Homing_B_Position_Ticks = b.Position_Ticks;
                    Homing_Hold_Pulses = std::llround(
                        static_cast<double>(Homing_B_Position_Ticks) * XY_Config::Pulses_Per_Rev / 65536.0);
                    Homing_Ack_Baseline = b.Position_Ack.Count;
                    Set_Phase(State::Home_Hold_B);
                }
            }
            break;
        case State::Home_Hold_B:
            result = Motor_B.Send_Position(Homing_Hold_Pulses, XY_Config::Homing_RPM,
                                           XY_Config::Acceleration, true, false);
            if (result == HAL_OK) Set_Phase(State::Home_Wait_Hold_B);
            break;
        case State::Home_Wait_Hold_B:
            if (b.Position_Ack.Count != Homing_Ack_Baseline)
            {
                if (b.Position_Ack.Status != 0x02U) Fail_Homing();
                else Set_Phase(State::Home_A);
            }
            break;
        case State::Home_A:
            Homing_Ack_Baseline = a.Homing_Ack.Count;
            Homing_Status_A_Baseline = a.Homing_Status_Count;
            result = Motor_A.Trigger_Homing(XY_Config::Driver_Homing_Mode, false);
            if (result == HAL_OK) Set_Phase(State::Home_Wait_A);
            break;
        case State::Home_Wait_A:
            if (a.Homing_Ack.Count != Homing_Ack_Baseline && a.Homing_Ack.Status != 0x02U)
            {
                Fail_Homing();
                break;
            }
            if (a.Homing_Status_Count != Homing_Status_A_Baseline)
            {
                Homing_Status_A_Baseline = a.Homing_Status_Count;
                if (a.Homing_Failed ||
                    (!a.Homing_In_Progress && a.Position_Valid && a.Speed_Valid &&
                     std::abs(a.Speed_RPM) <= XY_Config::Stopped_RPM))
                {
                    if (a.Homing_Failed) { Fail_Homing(); break; }
                    Homing_A_Position_Ticks = a.Position_Ticks;
                    Homing_Hold_Pulses = std::llround(
                        static_cast<double>(Homing_A_Position_Ticks) * XY_Config::Pulses_Per_Rev / 65536.0);
                    Homing_Ack_Baseline = a.Position_Ack.Count;
                    Set_Phase(State::Home_Hold_A);
                }
            }
            break;
        case State::Home_Hold_A:
            result = Motor_A.Send_Position(Homing_Hold_Pulses, XY_Config::Homing_RPM,
                                           XY_Config::Acceleration, true, false);
            if (result == HAL_OK) Set_Phase(State::Home_Wait_Hold_A);
            break;
        case State::Home_Wait_Hold_A:
            if (a.Position_Ack.Count != Homing_Ack_Baseline)
            {
                if (a.Position_Ack.Status != 0x02U) Fail_Homing();
                else Set_Phase(State::Home_Finish);
            }
            break;
        case State::Home_Finish:
            // 3B 只证明回零状态结束；最终原点仍以新鲜 36 多圈位置建立。
            if (Feedback_Fresh() && a.Position_Valid && b.Position_Valid &&
                a.Status_Valid && b.Status_Valid && a.Speed_Valid && b.Speed_Valid &&
                std::abs(a.Speed_RPM) <= XY_Config::Stopped_RPM &&
                std::abs(b.Speed_RPM) <= XY_Config::Stopped_RPM)
            {
                Origin_A = a.Position_Ticks;
                Origin_B = b.Position_Ticks;
                Origin_X = Now_X = XY_Config::Homing_Origin_X_mm;
                Origin_Y = Now_Y = XY_Config::Homing_Origin_Y_mm;
                Target_A = Origin_A;
                Target_B = Origin_B;
                Origin_Valid = true;
                Motion_Completed = true;
                Homing_Active = false;
                Homing_Failed = false;
                Fault = XY_Fault::None;
                Set_Phase(State::Idle);
            }
            break;
        default: Set_Fault(XY_Fault::Configuration); break;
    }
    if (result == HAL_ERROR) Set_Fault(XY_Fault::Transport);
    Poll();
}
