#include "2_Device/Motor/Motor_EmmV5/dvc_motor_emm_v5.h"
#include "3_Robot/xy_motion.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <algorithm>
#include <vector>

FDCAN_HandleTypeDef hfdcan1{1}, hfdcan2{2}, hfdcan3{3};
static uint32_t tick = 0;
static uint32_t tx_free = 8;
struct Frame { FDCAN_HandleTypeDef *bus; uint32_t id; std::vector<uint8_t> data; };
static std::vector<Frame> frames;
uint32_t HAL_GetTick() { return tick; }
uint32_t HAL_FDCAN_GetTxFifoFreeLevel(FDCAN_HandleTypeDef *) { return tx_free; }
HAL_StatusTypeDef CAN_Transmit_Extended_Data(FDCAN_HandleTypeDef *bus, uint32_t id,
                                           const uint8_t *data, uint8_t length) {
    if (tx_free == 0) return HAL_BUSY;
    frames.push_back({bus, id, std::vector<uint8_t>(data, data + length)});
    return HAL_OK;
}

static FDCAN_RxHeaderTypeDef header(uint8_t addr, uint32_t len) {
    FDCAN_RxHeaderTypeDef h{};
    h.Identifier = static_cast<uint32_t>(addr) << 8;
    h.IdType = FDCAN_EXTENDED_ID;
    h.DataLength = len;
    return h;
}
static bool rx(Class_Motor_EmmV5 &m, std::initializer_list<uint8_t> bytes) {
    std::vector<uint8_t> v(bytes);
    return m.CAN_RxCpltCallback(header(m.Get_Address(), v.size()), v.data());
}
static void pos(Class_XY_Motion &xy, uint8_t addr, int64_t ticks) {
    const uint32_t mag = static_cast<uint32_t>(ticks < 0 ? -ticks : ticks);
    const uint8_t data[] = {0x36, static_cast<uint8_t>(ticks < 0),
        static_cast<uint8_t>(mag >> 24), static_cast<uint8_t>(mag >> 16),
        static_cast<uint8_t>(mag >> 8), static_cast<uint8_t>(mag), 0x6B};
    xy.CAN_RxCpltCallback(header(addr, sizeof(data)), data);
}
static void speed(Class_XY_Motion &xy, uint8_t addr, uint16_t rpm) {
    const uint8_t data[] = {0x35, 0, static_cast<uint8_t>(rpm >> 8),
        static_cast<uint8_t>(rpm), 0x6B};
    xy.CAN_RxCpltCallback(header(addr, sizeof(data)), data);
}
static void status(Class_XY_Motion &xy, uint8_t addr, uint8_t flags = 3) {
    const uint8_t data[] = {0x3A, flags, 0x6B};
    xy.CAN_RxCpltCallback(header(addr, sizeof(data)), data);
}
static void feedback(Class_XY_Motion &xy, int64_t a = 0, int64_t b = 0,
                     uint16_t rpm = 0, uint8_t flags = 3) {
    pos(xy, XY_Config::Address_A, a); pos(xy, XY_Config::Address_B, b);
    speed(xy, XY_Config::Address_A, rpm); speed(xy, XY_Config::Address_B, rpm);
    status(xy, XY_Config::Address_A, flags); status(xy, XY_Config::Address_B, flags);
}
static void advance(Class_XY_Motion &xy, uint32_t ms = 1) {
    for (uint32_t i = 0; i < ms; ++i) { ++tick; xy.TIM_1ms_PeriodElapsedCallback(); }
}
static void expect_frame(size_t n, uint32_t id, std::initializer_list<uint8_t> bytes) {
    assert(n < frames.size());
    assert(frames[n].id == id);
    assert(frames[n].data == std::vector<uint8_t>(bytes));
}
static void test_protocol() {
    tick = 0; tx_free = 8; frames.clear();
    Class_Motor_EmmV5 m;
    m.Init(&hfdcan1, 1, 80);
    assert(m.Send_Position(-0x0001FA00, 4000, 0, false, false) == HAL_OK);
    expect_frame(0, 0x100, {0xFD, 1, 0x0F, 0xA0, 0, 0, 1, 0xFA});
    expect_frame(1, 0x101, {0xFD, 0, 0, 0, 0x6B});
    assert(m.Enable(true) == HAL_OK);
    expect_frame(2, 0x100, {0xF3, 0xAB, 1, 0, 0x6B});
    assert(m.Stop() == HAL_OK);
    expect_frame(3, 0x100, {0xFE, 0x98, 0, 0x6B});
    assert(Class_Motor_EmmV5::Sync(&hfdcan1) == HAL_OK);
    expect_frame(4, 0, {0xFF, 0x66, 0x6B});
    assert(m.Request_Encoder() == HAL_OK);
    expect_frame(5, 0x100, {0x31, 0x6B});
    assert(m.Request_Position() == HAL_OK);
    expect_frame(6, 0x100, {0x36, 0x6B});
    assert(m.Send_Position(std::numeric_limits<int64_t>::min(), 20, 0) == HAL_ERROR);
    assert(m.Send_Position(0x100000000LL, 20, 0) == HAL_ERROR);
    frames.clear(); tx_free = 1;
    assert(m.Send_Position(1, 20, 0) == HAL_BUSY);
    assert(frames.empty()); tx_free = 8;
    std::puts("PASS: exact CAN framing, signed bounds and atomic FIFO reservation");
}
static void test_feedback() {
    tick = 0; Class_Motor_EmmV5 m; m.Init(&hfdcan1, 1, 80);
    assert(rx(m, {0x36, 1, 0, 3, 0x80, 0, 0x6B}));
    assert(m.Get_Data().Position_Ticks == -229376);
    assert(rx(m, {0x32, 0, 0, 0, 0x0C, 0x80, 0x6B}));
    assert(m.Get_Data().Input_Pulses == 3200);
    assert(rx(m, {0x35, 1, 0, 120, 0x6B}));
    assert(m.Get_Data().Speed_RPM == -120);
    assert(rx(m, {0x3A, 0x0D, 0x6B}));
    assert(m.Get_Data().Enabled && m.Get_Data().Stall && m.Get_Data().Stall_Protection);
    assert(rx(m, {0xFD, 2, 0x6B}));
    assert(m.Get_Data().Position_Ack.Count == 1 && m.Get_Data().Position_Ack.Status == 2);
    assert(rx(m, {0xFD, 0x9F, 0x6B}));
    assert(m.Get_Data().Reached_Notification_Count == 1);
    const auto count = m.Get_Data().Position_Count;
    rx(m, {0x36, 0, 0, 0, 0, 1, 0});
    rx(m, {0x36, 2, 0, 0, 0, 1, 0x6B});
    assert(m.Get_Data().Position_Count == count);
    uint8_t data[] = {0x36, 0, 0, 0, 0, 1, 0x6B};
    auto h = header(2, 7);
    assert(!m.CAN_RxCpltCallback(h, data));
    h = header(1, 7); h.IdType = FDCAN_STANDARD_ID;
    assert(!m.CAN_RxCpltCallback(h, data));
    h = header(1, 7); h.Identifier |= 1;
    assert(!m.CAN_RxCpltCallback(h, data));
    assert(m.Get_Data().Position_Count == count);
    assert(rx(m, {0, 0xEE, 0x6B}));
    assert(m.Get_Data().Protocol_Error_Count == 1);
    std::puts("PASS: signed feedback, acknowledgements, malformed/unrelated frame rejection");
}
static void test_unwrap() {
    tick = 10; Class_Motor_EmmV5 m; m.Init(&hfdcan1, 1, 80);
    assert(rx(m, {0x31, 0xFF, 0xF0, 0x6B}));
    tick += 10; assert(rx(m, {0x31, 0, 0x10, 0x6B}));
    assert(m.Get_Data().Encoder_Unwrap_Valid && m.Get_Data().Software_Encoder_Ticks == 32);
    tick += 10; assert(rx(m, {0x31, 0xFF, 0xF0, 0x6B}));
    assert(m.Get_Data().Software_Encoder_Ticks == 0);
    tick += 81; rx(m, {0x31, 0, 0x10, 0x6B});
    assert(!m.Get_Data().Encoder_Unwrap_Valid && m.Get_Data().Unwrap_Lost_Count == 1);
    tick += 1; rx(m, {0x31, 0, 0x20, 0x6B});
    assert(!m.Get_Data().Encoder_Unwrap_Valid);
    rx(m, {0x36, 0, 0, 8, 0, 0, 0x6B});
    assert(m.Get_Data().Position_Valid && m.Get_Data().Position_Ticks == 524288);
    m.Reset_Encoder_Unwrap();
    rx(m, {0x31, 0, 0, 0x6B});
    ++tick; rx(m, {0x31, 0x80, 0, 0x6B});
    assert(!m.Get_Data().Encoder_Unwrap_Valid);
    tick = 0xFFFFFFF0U; m.Reset_Encoder_Unwrap();
    rx(m, {0x31, 0xFF, 0xF0, 0x6B});
    tick += 32; rx(m, {0x31, 0, 0x10, 0x6B});
    assert(m.Get_Data().Encoder_Unwrap_Valid && m.Get_Data().Software_Encoder_Ticks == 32);
    std::puts("PASS: positive/reverse wrap, dropped-sample and exact half-turn ambiguity");
}
static void test_matrix() {
    for (const auto &p : std::vector<std::pair<double, double>>{{0, 0}, {-77, 55}, {-20, 10}, {12, -34}}) {
        double a, b, x, y;
        Class_XY_Motion::XY_To_Motor_mm(p.first, p.second, a, b);
        Class_XY_Motion::Motor_To_XY_mm(a, b, x, y);
        assert(std::fabs(x - p.first) < 1e-9 && std::fabs(y - p.second) < 1e-9);
    }
    double a, b;
    Class_XY_Motion::XY_To_Motor_mm(-20, 10, a, b);
    assert(XY_Config::Frame_Rotation_Deg == 0.0);
    assert(std::fabs(a + 10.0) < 1e-9);
    assert(std::fabs(b + 30.0) < 1e-9);
    std::puts("PASS: XY transform inverse across workspace and signs");
}

// Emulate a CAN driver at the protocol boundary. Poll replies, control ACKs
// and measured motion are independently suppressible to exercise failure paths.
struct Rig {
    Class_XY_Motion xy;
    int64_t current[2] = {0, 0}, target[2] = {0, 0};
    int64_t pending_magnitude[2] = {0, 0};
    bool pending_negative[2] = {false, false};
    bool enabled[2] = {true, true};
    bool reply_feedback = true, drop_b_arm_ack = false, drop_sync_ack = false;
    bool hold_motion = false, running = false;
    uint8_t arm_result = 2, sync_result = 2, stop_result = 2;
    size_t processed = 0;
    unsigned sync_count = 0, stop_count = 0;
    uint32_t sync_tick = 0;
    Rig() {
        tick = 1000; tx_free = 8; frames.clear();
        assert(XY_Config::CAN_Bus() == &hfdcan2);
        xy.Init(); feedback(xy);
        assert(!xy.Get_Stop_Confirmed());
        assert(xy.Confirm_Origin_mm(0, 0) ==
            (XY_Config::Mechanical_Calibration_Confirmed ? HAL_BUSY : HAL_ERROR));
        step(220);
        assert(xy.Get_Stop_Confirmed());
        assert(stop_count == 2);
    }
    void ack(uint8_t addr, uint8_t function, uint8_t result = 2) {
        uint8_t data[] = {function, result, 0x6B};
        xy.CAN_RxCpltCallback(header(addr, sizeof(data)), data);
    }
    void process() {
        while (processed < frames.size()) {
            const Frame f = frames[processed++];
            // XY控制与轮询都必须走现场接线CAN2，不能只有配置声明改了总线。
            assert(f.bus == XY_Config::CAN_Bus());
            const uint8_t addr = static_cast<uint8_t>(f.id >> 8);
            const size_t axis = addr == XY_Config::Address_A ? 0 : 1;
            const uint8_t fn = f.data[0];
            if (fn == 0xFF) {
                ++sync_count; sync_tick = tick; running = true;
                if (!drop_sync_ack) ack(XY_Config::Address_A, fn, sync_result);
                continue;
            }
            if (fn == 0xF3) { enabled[axis] = f.data[2] != 0; ack(addr, fn); continue; }
            if (fn == 0xFE) { ++stop_count; running = false; ack(addr, fn, stop_result); continue; }
            if (fn == 0xFD) {
                if ((f.id & 0xFF) == 0) {
                    pending_negative[axis] = f.data[1] != 0;
                    pending_magnitude[axis] = (int64_t(f.data[5]) << 24) |
                        (int64_t(f.data[6]) << 16) | (int64_t(f.data[7]) << 8);
                } else {
                    const int64_t pulses = (pending_magnitude[axis] | f.data[1]) *
                        (pending_negative[axis] ? -1 : 1);
                    target[axis] = std::llround(pulses * 65536.0 / XY_Config::Pulses_Per_Rev);
                    if (!(drop_b_arm_ack && axis == 1)) ack(addr, fn, arm_result);
                }
                continue;
            }
            if (!reply_feedback) continue;
            if (fn == 0x36) pos(xy, addr, current[axis]);
            else if (fn == 0x35) speed(xy, addr, running && current[axis] != target[axis] ? 30 : 0);
            else if (fn == 0x3A) status(xy, addr,
                (enabled[axis] ? 1 : 0) | ((current[axis] == target[axis]) ? 2 : 0));
            else if (fn == 0x31) {
                const uint16_t encoder = static_cast<uint16_t>(current[axis]);
                const uint8_t data[] = {0x31, static_cast<uint8_t>(encoder >> 8),
                    static_cast<uint8_t>(encoder), 0x6B};
                xy.CAN_RxCpltCallback(header(addr, sizeof(data)), data);
            }
        }
    }
    void step(uint32_t ms = 1) {
        for (uint32_t n = 0; n < ms; ++n) {
            if (running && !hold_motion) {
                for (int axis = 0; axis < 2; ++axis) {
                    const int64_t difference = target[axis] - current[axis];
                    // 120 ticks/ms is below configured 300RPM maximum.
                    current[axis] += std::max<int64_t>(-120, std::min<int64_t>(120, difference));
                }
            }
            advance(xy); process();
        }
    }
    void origin(float x = 0, float y = 0) {
        assert(xy.Confirm_Origin_mm(x, y) == HAL_OK);
        assert(xy.Get_Position_Valid());
    }
};

static void test_origin_and_motion() {
    Rig r;
    assert(!r.xy.Get_Position_Valid());
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_ERROR);
    r.origin();
    assert(r.xy.Move_To_mm(-80, 10, 20, 20) == HAL_ERROR);
    assert(r.xy.Move_To_mm(-10, 10, 0, 20) == HAL_ERROR);
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    assert(!r.xy.Get_Motion_Finished());
    r.step(20);
    assert(r.sync_count == 1);
    assert(!r.xy.Get_Motion_Finished());
    r.step(1000);
    assert(!r.xy.Get_Fault() && r.xy.Get_Motion_Finished());
    assert(std::fabs(r.xy.Get_X_mm() + 10) < 0.02);
    assert(std::fabs(r.xy.Get_Y_mm() - 10) < 0.02);
    double a, b;
    Class_XY_Motion::XY_To_Motor_mm(-10, 10, a, b);
    assert(std::fabs(r.target[0]*XY_Config::Travel_Per_Rev_A_mm/65536.0 - a) < 0.01);
    assert(std::fabs(r.target[1]*XY_Config::Travel_Per_Rev_B_mm/65536.0 - b) < 0.01);
    // A stationary second move still needs new control ACKs and fresh feedback.
    const unsigned before_sync = r.sync_count;
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    r.step(30); assert(!r.xy.Get_Motion_Finished());
    r.step(200); assert(r.sync_count == before_sync + 1 && r.xy.Get_Motion_Finished());
    std::puts("PASS: origin interlock, target pulse scale, dual ACK sync and measured settle");
}

static void test_ack_timeout() {
    Rig r; r.origin(); r.drop_b_arm_ack = true;
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    r.step(700);
    assert(r.sync_count == 0);
    assert(r.xy.Get_Fault() && !r.xy.Get_Motion_Finished());
    assert(r.xy.Get_Fault_Code() == XY_Fault::Command_Timeout);
    assert(r.stop_count >= 2);
    assert(!r.xy.Get_Position_Valid());
    std::puts("PASS: missing arm ACK cannot release either axis; timeout stops both axes");
}

static void test_multi_turn_offset_origin() {
    Rig r; r.origin();
    assert(r.xy.Move_To_mm(-70, 20, 45, 45) == HAL_OK);
    r.step(2400);
    assert(r.xy.Get_Motion_Finished() && !r.xy.Get_Fault());
    assert(std::abs(r.current[0]) > 65536 && std::abs(r.current[1]) > 65536);
    assert(std::fabs(r.xy.Get_X_mm() + 70) < 0.02);
    assert(std::fabs(r.xy.Get_Y_mm() - 20) < 0.02);
    const int64_t base_a = r.current[0], base_b = r.current[1];
    // Confirm a known carriage coordinate while both motor encoders already
    // have multiple revolutions; neither absolute driver position is cleared.
    assert(r.xy.Confirm_Origin_mm(-60, 40) == HAL_OK);
    assert(r.xy.Move_To_mm(-50, 35, 20, 20) == HAL_OK);
    r.step(1200);
    assert(r.xy.Get_Motion_Finished() && !r.xy.Get_Fault());
    assert(std::fabs(r.xy.Get_X_mm() + 50) < 0.02);
    assert(std::fabs(r.xy.Get_Y_mm() - 35) < 0.02);
    double da, db;
    Class_XY_Motion::XY_To_Motor_mm(10, -5, da, db);
    assert(std::fabs((r.target[0]-base_a)*XY_Config::Travel_Per_Rev_A_mm/65536.0 - da) < 0.02);
    assert(std::fabs((r.target[1]-base_b)*XY_Config::Travel_Per_Rev_B_mm/65536.0 - db) < 0.02);
    std::puts("PASS: multi-turn workspace travel and known XY origin with nonzero motor counters");
}

static void test_feedback_loss_and_stop_retry() {
    Rig r; r.origin(); r.hold_motion = true;
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    r.step(30); assert(r.sync_count == 1);
    r.reply_feedback = false; r.step(200);
    assert(r.xy.Get_Fault() && r.xy.Get_Fault_Code() == XY_Fault::Feedback_Timeout);
    assert(!r.xy.Get_Motion_Finished() && !r.xy.Get_Position_Valid());
    r.reply_feedback = true; r.step(200);
    assert(r.xy.Get_Stop_Confirmed());
    assert(r.xy.Confirm_Origin_mm(0, 0) == HAL_OK);
    assert(!r.xy.Get_Fault());
    // 已完成的STOP可幂等复用；先开始新运动，再测TX BUSY时必须保持停机请求。
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    r.step(20);
    const unsigned before_stop = r.stop_count;
    // A full CAN TX FIFO must defer stop and preserve its pending state.
    tx_free = 0; r.xy.Stop(); r.step(20);
    assert(!r.xy.Get_Stop_Confirmed());
    assert(r.stop_count == before_stop);
    tx_free = 8; r.step(220);
    assert(r.xy.Get_Stop_Confirmed());
    assert(r.stop_count == before_stop + 2);
    assert(!r.xy.Get_Motion_Finished()); // cancelling a target is not arrival
    std::puts("PASS: feedback-loss latch, operator origin recovery and stop HAL_BUSY retry");
}

static void test_reset_jump_and_stall() {
    Rig r; r.origin();
    pos(r.xy, XY_Config::Address_A, 65536);
    r.step();
    assert(r.xy.Get_Fault_Code() == XY_Fault::Position_Jump);
    assert(!r.xy.Get_Position_Valid());
    Rig s; s.origin();
    status(s.xy, XY_Config::Address_A, 0x0D);
    s.step();
    assert(s.xy.Get_Fault_Code() == XY_Fault::Drive_Stall);
    std::puts("PASS: implausible encoder jump and drive stall stop the XY controller");
}

static void test_sync_and_command_errors() {
    Rig r; r.origin(); r.drop_sync_ack = true;
    assert(r.xy.Move_To_mm(-1, 1, 20, 20) == HAL_OK);
    r.step(700);
    assert(r.sync_count == 1 && r.xy.Get_Fault_Code() == XY_Fault::Command_Timeout);
    assert(!r.xy.Get_Motion_Finished());
    Rig s; s.origin(); s.arm_result = 0xE2;
    assert(s.xy.Move_To_mm(-1, 1, 20, 20) == HAL_OK);
    s.step(50);
    assert(s.sync_count == 0 && s.xy.Get_Fault_Code() == XY_Fault::Command_Rejected);
    Rig p; p.origin(); p.drop_b_arm_ack = true;
    assert(p.xy.Move_To_mm(-1, 1, 20, 20) == HAL_OK);
    p.step(20);
    p.ack(XY_Config::Address_B, 0, 0xEE);
    p.step();
    assert(p.xy.Get_Fault_Code() == XY_Fault::Command_Rejected);
    std::puts("PASS: missing sync ACK, rejected arm and protocol error cannot report motion complete");
}

static void test_disabled_stop_ack() {
    Rig r;
    // 从已失能的安全状态确认原点，单独检验E2停机规则，不混入运行后失能故障。
    r.enabled[0] = r.enabled[1] = false;
    r.step(20); r.origin(); r.hold_motion = true;
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    // 两轴使能控制已实际下发，但还没有进入Moving；这次STOP必须重新发FE。
    r.step(4);
    assert(r.enabled[0] && r.enabled[1]);
    const unsigned before_stop = r.stop_count;
    r.stop_result = 0xE2;
    r.xy.Stop(); r.step(220);
    assert(r.stop_count == before_stop + 2);
    assert(!r.xy.Get_Stop_Confirmed());
    r.enabled[0] = r.enabled[1] = false;
    r.step(220);
    assert(r.xy.Get_Stop_Confirmed());
    std::puts("PASS: FE/E2 accepted only with new disabled and stationary feedback");
}

static void test_repeated_stop_preserves_completed_confirmation() {
    Rig r;
    const unsigned startup_stops = r.stop_count;
    assert(startup_stops == 2); // 上电仍须向两台电机实际发送STOP。
    r.origin();
    assert(r.xy.Get_Motion_Finished());
    for (unsigned n = 0; n < 6; ++n) {
        r.xy.Stop();
        assert(r.xy.Get_Stop_Confirmed());
        assert(!r.xy.Get_Motion_Finished()); // 重发STOP仍取消旧到位事件，但不重置停稳确认。
        r.step(250); // 与Jetson启动同步的重复周期一致。
        assert(r.xy.Get_Stop_Confirmed());
        assert(r.stop_count == startup_stops);
    }

    // 新运动撤销旧STOP确认，必须重新发送双轴停机并等待真实反馈稳定。
    r.origin(); r.hold_motion = true;
    assert(r.xy.Move_To_mm(-10, 10, 20, 20) == HAL_OK);
    r.step(20);
    r.xy.Stop();
    assert(!r.xy.Get_Stop_Confirmed());
    r.step(220);
    assert(r.xy.Get_Stop_Confirmed());
    assert(r.stop_count == startup_stops + 2);
    assert(!r.xy.Get_Motion_Finished());
    std::puts("PASS: repeated startup STOP retains safe confirmation; new movement requires a new stop");
}

static void test_stale_feedback_does_not_reuse_completed_stop() {
    Rig r;
    const unsigned before_stop = r.stop_count;
    r.reply_feedback = false;
    r.step(XY_Config::Feedback_Timeout_ms + 20);
    assert(!r.xy.Get_Stop_Confirmed());
    r.xy.Stop();
    r.step(20);
    assert(r.stop_count == before_stop + 2);
    assert(!r.xy.Get_Stop_Confirmed()); // 新STOP ACK也不能代替新鲜位置/速度/状态。
    r.reply_feedback = true;
    r.step(220);
    assert(r.xy.Get_Stop_Confirmed());
    std::puts("PASS: stale feedback invalidates repeated-stop reuse until fresh feedback settles");
}

int main() {
    if (!XY_Config::Mechanical_Calibration_Confirmed) {
        Rig r;
        assert(r.xy.Get_Stop_Confirmed());
        assert(r.xy.Confirm_Origin_mm(0, 0) == HAL_ERROR);
        assert(r.xy.Move_To_mm(-1, 1, 20, 20) == HAL_ERROR);
        assert(!r.xy.Get_Position_Valid());
        std::puts("PASS: shipped uncalibrated configuration permits feedback/stop but prohibits origin and motion");
        return 0;
    }
    test_protocol(); test_feedback(); test_unwrap(); test_matrix();
    test_origin_and_motion(); test_ack_timeout();
    test_multi_turn_offset_origin();
    test_feedback_loss_and_stop_retry(); test_reset_jump_and_stall();
    test_sync_and_command_errors(); test_disabled_stop_ack();
    test_repeated_stop_preserves_completed_confirmation();
    test_stale_feedback_does_not_reuse_completed_stop();
    std::puts("All host regressions passed.");
}
