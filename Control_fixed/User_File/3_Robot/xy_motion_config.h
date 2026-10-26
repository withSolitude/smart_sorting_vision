#ifndef XY_MOTION_CONFIG_H
#define XY_MOTION_CONFIG_H

#include "fdcan.h"
#include <cstdint>

// 所有待实机标定参数集中在这里。未确认标定前只读反馈，禁止运动。
namespace XY_Config
{
// 两台 EMM 步进驱动器接在 CAN2；CAN1 保留给原有 DM4310 总线。
inline FDCAN_HandleTypeDef *CAN_Bus() { return &hfdcan2; }
constexpr uint8_t Address_A = 1U;
constexpr uint8_t Address_B = 2U;
constexpr uint32_t CAN_Baud = 1000000U; // CAN2、两台 EMM 和 TTL-CAN 适配器统一 1 Mbps
#if defined(EMM_V5_HOST_TEST) && defined(XY_TEST_CALIBRATED)
constexpr bool Mechanical_Calibration_Confirmed = true; // 仅主机回归测试
#else
constexpr bool Mechanical_Calibration_Confirmed = true; // 实机标定后改 true
#endif

// +45° 表示机构 U 轴相对视觉 +X 逆时针旋转；顺时针请改 -45°。
constexpr double Frame_Rotation_Deg = 0.0;
constexpr double Pi = 3.14159265358979323846;

// CoreXY: A=U+V, B=U-V。正的电机坐标对应协议 CW(方向0)。
// 换向时只改这里，不能在视觉侧和电控侧重复换向。
constexpr int Direction_A = 1;
constexpr int Direction_B = 1;
constexpr double Travel_Per_Rev_A_mm = Pi * 12.0; // 待标定！不是已知硬件参数
constexpr double Travel_Per_Rev_B_mm = Pi * 12.0; // 带轮齿数*带距/减速比
constexpr uint16_t Full_Steps_Per_Rev = 200U; // MotType=1.8deg
constexpr uint16_t Microsteps = 16U;         // 电机 MStep 必须一致
constexpr uint32_t Pulses_Per_Rev = Full_Steps_Per_Rev * Microsteps;
constexpr uint16_t Max_RPM = 300U;
constexpr uint8_t Acceleration = 0U; // 0=恒速；两轴按位移比例分配转速

// 沿用当前视觉协议工作区；新结构调试时必须重新测量。
constexpr float X_Min_mm = -770.0f;
constexpr float X_Max_mm = 700.0f;
constexpr float Y_Min_mm = -800.0f;
constexpr float Y_Max_mm = 550.0f;
constexpr double Position_Tolerance_mm = 0.30;
constexpr double Stopped_RPM = 2.0;
constexpr uint32_t Feedback_Timeout_ms = 150U;
constexpr uint32_t Command_Timeout_ms = 500U;
constexpr uint32_t Move_Timeout_ms = 15000U;
constexpr uint32_t Settle_ms = 150U;
constexpr uint32_t Poll_Interval_ms = 2U; // 每16ms读取一次每台电机的31/36/35/3A
constexpr uint16_t Encoder_Unwrap_Max_Gap_ms = 80U; // 300RPM 时半圈=100ms

// 自动回零默认关闭。只有确认 CoreXY 机械硬挡位置、方向和驱动器碰撞参数后才改为 true。
// 当前 +45° 关系为 A≈√2·Y、B≈√2·X，回零必须先单独回 B(X)，再单独回 A(Y)。
constexpr bool Driver_Collision_Homing_Enabled = false;
constexpr uint8_t Driver_Homing_Mode = 2U; // 2=多圈无开关碰撞回零
constexpr uint16_t Homing_RPM = 30U; // 首次调试低速，实际以驱动器 O_Vel 为准
constexpr uint32_t Homing_Timeout_ms = 20000U;
constexpr uint32_t Homing_Status_Poll_ms = 20U;
constexpr float Homing_Origin_X_mm = 0.0f;
constexpr float Homing_Origin_Y_mm = 0.0f;

// 上电置零：两台电机反馈首次同时有效、停稳且没有堵转后，
// 把“上电时的当前位置”记录为软件 XY 原点。
// 这不是碰撞回零，也不会让电机自动运动；上电位置必须由机械人员先放到零点。
#if defined(EMM_V5_HOST_TEST)
// 旧主机回归用例显式测试“人工确认原点”互锁，避免自动置零改变测试前置条件。
constexpr bool Power_On_Origin_Enabled = false;
#else
constexpr bool Power_On_Origin_Enabled = true;
#endif
constexpr float Power_On_Origin_X_mm = 0.0f;
constexpr float Power_On_Origin_Y_mm = 0.0f;
}

#endif
