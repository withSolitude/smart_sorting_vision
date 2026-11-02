#ifndef EDGE_MOTION_CONFIG_H
#define EDGE_MOTION_CONFIG_H

// 危险边界：靠近电机/相机的一侧。所有 XY/Yaw 联动统一在此边界外完成转向。
// 边界"在哪条轴上"由 Boundary_Axis_Y 决定，两条轴共用同一个边界数值。
//
// 现场实测（待机位在图像右下角）：
//   电机与相机在图像的**下边沿** -> 危险轴是 Y，X 是沿边沿行进的方向。
// 机构重新装配/换向时只改这一个开关：
//   true  = 边界在 Y（y=0 是下边沿）
//   false = 边界在 X（x=0 是右边缘，旧设计）
// 上位机 edge_yaw_limit.boundary_axis 必须与这里一致。
namespace Edge_Motion_Config
{
constexpr bool Boundary_Axis_Y = true;
constexpr float Turn_Boundary_mm = -40.0f;
constexpr float Turn_Extra_Clearance_mm = 0.01f; // 再向内留少量数值余量，叠加XY到位容差。

// 兼容旧名：只表示"边界数值"，不再表示边界一定在 X 轴上。
constexpr float Turn_Boundary_X_mm = Turn_Boundary_mm;
}

#endif
