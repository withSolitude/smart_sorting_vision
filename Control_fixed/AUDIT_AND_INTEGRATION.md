# Control 下位机严格审查与 XY 标定整合说明

> 2026-10-24 更新：下面涉及 XY 脉冲驱动及“上电位置作为零点”的结论已被 Emm V5 CAN 迁移替代。当前接入、配置、位置有效性与实机验收以 [EMM_V5_XY_MIGRATION.md](EMM_V5_XY_MIGRATION.md) 为准。旧记录保留供追溯。

## 本次已整合

- Jetson -> STM32 `CALIB_MOVE = 0x03`
- STM32 -> Jetson `MCU_CALIB_MOVE_ACK = 0x89`
- STM32 -> Jetson `MCU_CALIB_REACHED = 0x8A`
- 复用 `MCU_ERROR = 0xE0`
- 标定工作区：X = -77.0 ~ 0.0 mm，Y = 0.0 ~ 55.0 mm
- 标定模式只移动 X/Y，不执行抓取、夹爪、储物投放
- X/Y 均停止且误差 <= 0.30 mm，连续稳定 150 ms 后才发送 REACHED
- 标定运动超时 15 s
- 标定期间实体按键仍可 STOP

## 已修复的高优先级问题

### P0-1 储物机构 ERROR 不会上抛
原逻辑中 `Sorting_Storage` 进入 `Sorting_Storage_Status_ERROR` 后，Robot 层仍可能保持 `Sort_Action_Active=true`，Jetson 会永久等待 `SORT_DONE`。

修复：Robot 10 ms 调度检测 Storage ERROR，清除待执行任务，通知握手层 `MCU_ERROR`，并请求储物机构执行安全回位。

### P0-2 UART DMA 发送缓冲区可能被下一帧覆盖
原 `Send_Frame()` 使用同一个 `Tx_Frame_Buffer` 做 DMA 源，但在确认 UART READY 之前就改写缓冲区。若上一帧 DMA 尚未结束，下一次发送即使最终返回 HAL_BUSY，也已经可能破坏正在发送的帧。

修复：短临界区内先检查 `UART_Handler->gState == HAL_UART_STATE_READY`，只有 READY 才填充共享 TX buffer 并启动 DMA。

### P0-3 上电自动 Start
`Task_Init()` 最后无条件调用 `robot.Start_Sorting_Work()`，绕过实体按键逻辑，也会让独立 XY 标定命令无法在 STOPPED 状态进入。

修复：删除自动 Start。正式流程由 PC1 按键启动；STOPPED 状态可接受 CALIB_MOVE。

### P0-4 TOF 下降状态可能永久卡死
原逻辑 TOF 长时间无新数据时只 `return`，`Grap_Move()` 又只等待 `Grap_TOF_Reached=true`，会永久卡在 `Enum_Grap_Move_Descend`。若 TOF 始终在线但距离一直大于阈值，Lift 到机械下降极限后也会永久等待。

修复：TOF 超时或下降到机械极限仍未命中时，设置 `Grap_Failed=true`，机构转入 Return 安全回位。Robot 完成回位后发送 ERROR，不再错误发送 RECOGNITION_READY。

### P1-1 PICK 启动 HAL_BUSY 无超时
原逻辑 `Start_Grap()` 返回 HAL_BUSY 时无限重试。

修复：增加 3 s 启动等待超时。

### P1-2 SORT 启动只尝试一次
原逻辑一旦 `Sorting_Storage.Start_Sort()` 返回 BUSY 就直接 ERROR，容易被 DM4310/UART 瞬态状态误触发。

修复：保存 Pending Sort Request，允许最多 3 s 的 BUSY 重试；HAL_ERROR 或超时才进入 ERROR。

### P1-3 下位机缺少 X/Y 硬限位二次防护
原 `Start_Grap()` 本身不检查视觉目标机械工作区。

修复：下位机再次检查 X∈[-77,0]、Y∈[0,55]，并检查 finite 值。上位机限位 + 下位机限位形成双保险。

### P1-4 储物角度输入未归一化
原 `Set_Box_Angle()` 可写入任意角度；如果超出合理范围，`Get_Nearest_Equivalent_Angle()` 可能找不到合法等效角，随后底层再静默 clip，最终可能转到错误箱位。

修复：所有箱位基础角与 Disable 基础角统一归一化到 [0, 2π)，再根据当前真实角度计算最接近的等效圈数，并检查 DM P_MAX 范围。

### P2-1 Task 中存在第二个未初始化 TOF 对象
原 `tsk_config_and_callback.cpp` 有一个全局 `Class_TOF050F TOF050F`，调试变量读取的是这个未初始化实例，而实际工作的实例是 `robot.Holding_Jaw.TOF050F`。

修复：调试数据改为读取真实工作实例。

## 储物/分箱逻辑结论

下位机当前 `sorting_storage` 是“物理投放机构状态机”，不是库存数据库。它负责：4310 转箱 -> 托盘翻转 -> 保持 -> 回位 -> 4310 回待机 -> `Sort_Finished_Pending`。

库存数量、容量、reserve/commit/cancel 应继续由 Jetson Step 1 的 StorageManager 管理：收到 SORT_RESULT 前 reserve，只有 MCU_SORT_DONE 且 Box_ID 匹配后 commit；ERROR/STOP/超时 cancel。不要在 STM32 再维护第二套库存计数，否则上下位机会产生双真源问题。

目前下位机已经做到：
- Box_ID 仅接受 1~6；
- 重复 SORT_RESULT 在执行中只重发 ACK，不重复启动机械；
- 完成后若 SORT_DONE 丢失，Jetson 重发同一 SORT_RESULT 时可重发 DONE；
- Storage 内部 ERROR 会向上位机传播；
- Start_Sort 的 BUSY 不再立即丢任务。

## 4310 旋转角度逻辑结论

六箱基础角默认仍是 0/60/120/180/240/300 deg。程序使用的是 rad。

本次修订后：
1. 基础角先归一化到 [0, 2π)；
2. 读取 DM4310 当前真实 `Now_Angle`；
3. 估计最接近当前角度的整数圈 k；
4. 在相邻圈中寻找仍处于 `[-Angle_Max, +Angle_Max]` 的最短候选；
5. 到位判据同时使用角度误差 <= 2 deg 和 |omega| <= 0.15 rad/s，并连续 5 个 10 ms 周期满足。

因此软件层面的“240/300 度跨界绕远路”问题已得到处理。

仍需实机确认：如果 4310 转盘存在电缆/软管，机械上不允许多圈旋转，那么仅依赖 P_MAX=±12.5 rad 仍不够，应再设置一个更窄的物理安全角范围；当前代码无法从工程文件判断你的转盘是否允许连续多圈。

## 仍未自动修复、必须实机确认的问题

### R1：X/Y 没有真实回零（最高优先级）
`Holding_Jaw.Init()` 仍然直接：

```cpp
X_Stepping_Motor.Set_Origin_mm(0.0f);
Y_Stepping_Motor.Set_Origin_mm(0.0f);
```

这只是“把上电瞬间所在位置定义成 0”，不是机械回零。如果每次上电夹爪不严格处在真实 (0,0)，那么所有绝对 X/Y 坐标都会整体偏移，XY 标定也会失效。

在没有看到 X/Y 原点限位开关/光电开关接口前，本次没有虚构回零流程。正式运行前建议增加：回零开关 -> 慢速寻零 -> 退回 -> 二次慢速触发 -> Set_Origin_mm(0)。

### R2：步进轴没有闭环位置反馈
`Get_Position_mm()` 是按已输出脉冲累计的软件位置，不代表机械一定实际到位。堵转、丢步、同步带打滑时软件仍会认为 reached。

因此本次 `MCU_CALIB_REACHED` 表示“脉冲运动完成且软件位置稳定”，不是编码器闭环实测位置。若需要更高安全性，应增加原点/末端限位，最好增加编码器或位置传感器。

### R3：储物箱实际角度仍需逐箱标定
默认 60° 等分只是几何假设。装配误差、零位偏差、托盘落料中心都会使实际最佳角度偏离。应实机逐箱设置 `Set_Box_Angle()`。

### R4：托盘舵机 PWM 仍需实机验收
当前 Disable=1580，Flip=900，运动时间 400 ms + 100 ms safety，Hold=500 ms。软件逻辑无明显冲突，但必须观察是否真的完全翻转/复位，不能只按延时判断机械完成。

## 建议实机验证顺序

1. 不装物体，确认上电后不再自动 START。
2. 手动确保当前机械确实在 X=0,Y=0（在正式回零未加入前）。
3. 使用 Jetson XY 标定程序先发 (-5,5)，确认方向正确。
4. 再测试 (-72,5)、(-72,50)、(-5,50)，确认不会越机械边界。
5. 测试运动中按 PC1，确认 X/Y 标定可立即 STOP。
6. 测试一个完整 PICK -> READY，不放物体时验证 TOF timeout 会 ERROR 而不是永久卡住。
7. 单独测试 1~6 箱，记录 4310 实际最短旋转方向和最终落料位置。
8. 人为断开/禁用 4310，确认 MCU_ERROR 可传到 Jetson，且 Jetson 库存 reserve 会 cancel。

## 编译说明

当前容器没有 `arm-none-eabi-gcc/g++`，因此无法在本环境完成 ARM 固件链接验证。已完成修改文件的结构/括号检查和接口级静态审查。请在你原来的 CLion/ARM GCC 环境重新 CMake configure 后编译，不能复用压缩包里的旧 `cmake-build-debug/CMakeCache.txt`。
