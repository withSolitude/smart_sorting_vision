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
// 现场实测：退避/中转点现在是 (x=0, y=-60)，仍在 Y 轴上 -> Boundary_Axis_Y = true。
// 上电日志会打印实际退避点，若与现场不符请先核对这一项。
constexpr bool Boundary_Axis_Y = true;
constexpr float Turn_Boundary_mm = -60.0f;
constexpr float Turn_Extra_Clearance_mm = 0.01f;

// true = 从危险带内出发时不再"先退到边界点再横移"，而是直接一条斜线走向目标；
//        Yaw 会延迟到实测离开危险带之后再发（见 Move_XY_Yaw_Via_Edge 的 step 4），
//        所以"危险带内不转 Yaw"这个安全不变量仍然成立。
// 代价：斜线在 y 从 0 走到边界的那一段，X 已经开始移动。若夹爪在 y≈0 附近
//       横移会碰到电机/龙门梁，请保持 false 或先把 Turn_Boundary_mm 收紧。
constexpr bool Merge_Start_With_Travel = true;

// 图片右边沿（x 接近 0）到识别托盘的连接处是竖直面，而夹爪抬升余量有限：
// 夹着高件贴着右边沿往 y=0 走，下面只要有一个稍高的物体就会卡死。
// 所以物体本身落在 x > Right_Edge_Keepout_X_mm 时，放件前先横向退到
// Right_Edge_Detour_X_mm，再从那里直接去识别托盘。
constexpr bool Right_Edge_Detour_Enabled = false;  // 比赛版先关掉：开着时在 x=-40 处会卡住，待后续修。
constexpr float Right_Edge_Keepout_X_mm = -30.0f;
constexpr float Right_Edge_Detour_X_mm = -40.0f; // 再向内留少量数值余量，叠加XY到位容差。

// 兼容旧名：只表示"边界数值"，不再表示边界一定在 X 轴上。
constexpr float Turn_Boundary_X_mm = Turn_Boundary_mm;
}

#endif
