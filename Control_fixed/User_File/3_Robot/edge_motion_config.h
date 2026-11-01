#ifndef EDGE_MOTION_CONFIG_H
#define EDGE_MOTION_CONFIG_H

// x=0 一侧有电机和相机。所有 XY/Yaw 联动统一在此边界外转向。
// 上位机的抓取候选区域配置应使用同一个实测边界值。
namespace Edge_Motion_Config
{
constexpr float Turn_Boundary_X_mm = -40.0f;
constexpr float Turn_Extra_Clearance_mm = 0.01f; // 再向内留少量数值余量，叠加XY到位容差。
}

#endif
