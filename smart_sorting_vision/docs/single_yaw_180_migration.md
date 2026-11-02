# 单 180° 抓取与 Yaw 参数同步

抓取闭合轴仍由顶部视觉输出，协议角度保持 `[-90°, 90°)`，精度为 1°。
不再搜索第二个物理姿态，也不再发送 `PICK_YAW_BRANCH` 或等待其 ACK。
每件物体的发送顺序为 `PICK_PREOPEN → ACK → PICK_TARGET → ACK → RECOGNITION_READY`。

## 唯一物理姿态

上下位机使用相同公式：

```text
delta = ((axis - theta0 + 90) mod 180) - 90
physical = theta0 + delta
servo = q0 + k * delta
```

`mod` 的结果必须非负。物理抓取范围为 `[theta0-90, theta0+90)`。
相差 180° 的闭合轴得到同一条舵机命令。该映射也用于 TCP 偏心和下降漂移补偿。
视觉规划先量化协议角度，再计算物理姿态，避免窗口边界附近出现 180° 错配。

2026-10-30 当前标定参数（与上传 YAML 和 MCU 一致）：

| 参数 | 数值 |
|---|---:|
| theta0 | -63.280987° |
| q0 | 0.232430° |
| k | 1.04720792 |

因此物理区间约为 `[-153.2810°, 26.7190°)`，舵机命令约在 `按实际安全限位裁剪为 `[-90°, 90°]``。
此前报错的 `axis=-82°` 现在输出 `servo≈-19.3703°`，不会再按旧参数算成 `+137.78°`。
这只是公式和软件限位验证；五点报告实际覆盖的舵机命令为 -60° 到 +60°，全行程仍需空载验证。

## 配置位置

上位机只读取 `config/runtime_config.yaml` 的 `calibration.yaw_servo`。
下位机对应 `Control_fixed/User_File/4_Interaction/ita_robot.h` 的三个 `Pick_Yaw_*` 标定值。
两端舵机安全限位保持 `[-90°, 90°]`，TCP 角度范围从该 Yaw 标定自动导出。
旧的 branch1 修正和 TCP min_axis_deg/max_axis_deg 配置已移除。

预开口仍以毫米发送，保持已实测的 `13～77mm` 范围和 `5mm` 总余量。
没有可靠动态预开口时沿用下位机固定预收缩角，仍直接发送 PICK_TARGET；有动态值时先发送 PICK_PREOPEN。

## 更新与验证

1. 上下位机配套更新。重新编译、烧录 MCU，并更新 Jetson 运行目录中的源码和 YAML。
2. 启动日志应出现 `[CONFIG] 单 180° Yaw`；抓取日志应出现 `single-yaw selected`。
3. 串口应直接从 `PICK PREOPEN ACK` 进入 `PICK_TARGET`，没有 branch 选择事务。
4. 单步空载验证 `-82°、-45°、0°、45°、89°`，先检查指向与 TCP 对心，再进行低速抓取。
5. 核对预开口 `21.6mm≈-24.53°`；区分下降前的动态预开口和下降后的正式夹紧。

不应在修改 Yaw 标定时重调 XY、ROI 或预开口公式来抵消方向误差。
