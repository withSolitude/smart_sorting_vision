# Emm V5 上位机、TTL-CAN 与 STM32 调试设置

当前工程中两台 EMM 固定使用 **STM32 CAN2、1 Mbps、经典 CAN、扩展帧**。CAN1 保留给原有 DM4310 总线。上位机下拉框可以选择其他波特率，但 STM32 CAN2 位时序、两台 EMM 和 TTL-CAN 必须同时一致；当前不要只把上位机改成 500 kHz。

TTL-CAN 适配器有两个不同的速率：

1. **TTL 串口侧速率**：由适配器说明书决定；如果适配器默认是 115200 8N1，就在串口工具中保持 115200。
2. **CAN 总线速率**：截图中的 CAN 速率，当前设为 **1000000**。它必须与 STM32 CAN2 和 Emm 菜单里的 CAN_Baud 一致。

TTL 串口的 TX/RX 不能直接接 CANH/CANL。CANH/CANL 还需要正确的收发器、共地和总线两端 120 Ω 终端电阻。STM32 CAN2、两台 EMM 和 TTL-CAN 挂在 EMM 总线上；DM4310 仍使用 CAN1。

## 上位机截图中的推荐设置

每台 Emm 分别连接和读取一次。先设置地址 1 的 A 电机，保存并重新上电，再设置地址 2 的 B 电机。

| 上位机项目 | 推荐值 | 说明 |
|---|---:|---|
| 端口 | TTL-CAN 实际 COM 口 | 不是任意串口；先在设备管理器确认 |
| CAN 速率 | **1000000** | 当前 STM32 CAN2 为 1 Mbps |
| 电机地址 | A=1，B=2 | 地址不能重复；广播同步要求至少有地址 1 |
| 校验字节 | 0x6B | 与代码固定校验一致 |
| 电机类型 | 1.8° | 与代码 `Full_Steps_Per_Rev=200` 一致 |
| 脉冲控制模式 | FOC | 不要选开环 PUL_OPEN |
| 通讯端口复用 | CAN1_MAP | 这是 EMM 驱动器菜单中的 CAN 通道映射名称，需与适配器接线一致 |
| En 有效电平 | Hold | 由 CAN 控制使能，避免 En 悬空 |
| 细分 | **16** | 代码 `Microsteps=16`；截图中的 15 要改掉 |
| 细分插补 | Enable | 降低低速噪声，可按实机关闭比较 |
| 串口波特率 | 115200 | 仅影响驱动器串口侧，不是 CAN 总线速率 |
| CAN 波特率 | 1000000 | 驱动器内部参数必须与总线一致 |
| 通讯校验方式 | 0x6B | 不要选 XOR/CRC/Modbus |
| 控制命令应答 | Receive | 代码依赖命令接收 ACK；调试时也可选 Both |
| 速度缩小 10 倍 | Disable | 代码速度单位按整数 RPM 发送 |
| 上电自动回零 | Disable | 首次调试禁止上电自动撞机械挡 |
| 堵转保护 | 初次方向/尺度调试可 Disable | 调好碰撞回零阈值后再 Enable |

`Dir` 不能凭截图直接确定。先用 10~30 RPM、小脉冲测试，记录驱动器 CW/CCW 对应的机械方向，再修改 `xy_motion_config.h` 中 `Direction_A`、`Direction_B`。代码和上位机的方向不能重复取反。

## 位置模式中的几个下拉框

上位机“位置模式”中的“相对/绝对标志”只影响上位机手动发出的那一条位置命令。当前 STM32 代码在 `Send_Position()` 中明确发送：

- 普通运动：绝对位置标志 `absolute=true`；
- 两轴协同运动：同步标志 `sync=true`，两台驱动都收到后由代码发送 `0xFF` 广播同步；
- 停止：发送 `0xFE 0x98`；
- 查询：周期读取 0x36 位置、0x35 速度、0x3A 状态、0x31 单圈编码器。

因此上位机手动测试时可选“绝对运动”、同步标志 False；正式 CoreXY 运动由 STM32 自己下发，不依赖上位机当前表单值。

## 驱动器编码器校准后，直接用代码测试的顺序

1. 断开皮带或让机构处于安全空载状态，完成两台驱动器的编码器校准；确认没有 `Not Cal`、`Magnet Loss`、线序错误或电压错误。
2. 在上位机分别读取 A/B 的系统状态，确认两台扩展 CAN 地址、1 Mbps、0x6B、CAN 通道映射和 16 细分正确。
3. 先只观察代码反馈，不运动：检查 `Position_Ticks`、`Encoder`、`Speed_RPM`、`Status_Flags` 和各时间戳是否持续刷新。代码启动后会先给两台驱动发送停止，避免 MCU 重启后旧目标继续运动。
4. 将 `xy_motion_config.h` 中的 `Travel_Per_Rev_A_mm/B_mm` 填成实际每转带长，测出 `Direction_A/B`，再确认 `Frame_Rotation_Deg=45`。只完成驱动器编码器校准，还不能把 40 mm 占位值当成真实机械参数。
5. 确认机械参数后，把 `Mechanical_Calibration_Confirmed` 改为 true 并重新编译。它是绝对 XY 运动的安全锁。
6. 初次使用代码时保持 `Driver_Collision_Homing_Enabled=false`。每次上电前先把滑台放到机械零点；代码会在 CAN2 的两台电机位置、速度、状态反馈都新鲜且停稳后，把“上电时当前位置”自动记录为软件原点 `(0,0)`。这不会发送清除多圈位置命令，也不会自动移动电机。若需要手动覆盖原点，仍可写：

```text
XY_Origin_Confirm_X_mm = 0
XY_Origin_Confirm_Y_mm = 0
XY_Origin_Confirm_Request = 1
```

若机械零点不是 `(0,0)`，写实际的视觉坐标。可观察 `Get_Power_On_Origin_Captured()` 确认自动置零完成；机械参数确认后且 `Get_Position_Valid()` 为真时才允许绝对 XY 运动。

7. 先测试很小的目标，例如 `(x,y)=(-5,5)`，再测试 `(-20,10)`，最后才测试工作区边界。每次只改变一个变量，按 PC1 长按或代码 STOP 验证停车。
8. 正常代码运动时观察 A≈√2·Y、B≈√2·X 的关系（包含方向符号），不要把 A 直接当 X、B 直接当 Y。

## 碰撞回零怎么设置

上位机“原点回零参数”可以选择：

| 项目 | 首次建议 |
|---|---:|
| 回零模式 | Senless（多圈无开关碰撞） |
| 回零方向 | 每台单独低速试方向，不能直接假设 A/B 相同 |
| 回零速度 | 30 RPM 起步 |
| 回零超时 | 10000~20000 ms |
| 无限位检测转速 | 30~60 RPM 起步 |
| 无限位检测电流 | 先测正常电流，再从正常值的约 1.3~1.5 倍起步 |
| 无限位检测时间 | 60~100 ms 起步 |
| 上电自动回零 | Disable |

说明书的 800 mA、300 RPM、60 ms 只是示例，不能直接套到新电机。碰撞回零的判定是“速度低于阈值 + 相电流高于阈值 + 持续时间超过阈值”。阈值太低会误触发，太高会硬顶机械挡。

CoreXY 不能让 A/B 同时自由碰撞。当前 45° 变换下：

1. 先让 **B 单独回 X 方向机械挡**，A 保持使能和位置；
2. B 完成后保持 B 在原点，再让 **A 单独回 Y 方向机械挡**；
3. 两台都完成后读取新的 0x36 多圈位置，把它们写入 STM32 软件原点；不默认发送 0x0A 清除驱动器多圈位置。

代码已提供 0x9A（触发回零）、0x3B（读取回零状态）、0x9C（中断回零）接口，顺序状态机默认由 `Driver_Collision_Homing_Enabled=false` 锁住。首次运行必须先人工确认方向、电流和机械挡的风险，再打开这个配置。

## 需要重点观察的调试量

- **Emm A/B**：扩展帧 ID、ACK 状态（02 成功、E2 条件不满足、EE 错误）、0x36 多圈位置及时间戳、0x31 单圈编码器和软件跨圈有效标志、0x35 RPM、0x3A 状态位、0x3B 回零标志、坏帧/协议错误/跨圈丢失计数。
- **XY**：`Now_X/Now_Y`、`Get_Position_Valid()`、`Get_Stop_Confirmed()`、`Get_Motion_Finished()`、目标 A/B 位置、A/B 速度、当前故障码、回零是否进行中/失败。
- **储物机构**：DM4310 输出轴角度、角速度、反馈时间戳、0x51 更新时间、目标角度、PID 力矩和故障锁存。
- **夹爪**：抓取状态、错误码、Disable/Stop Ready、TOF 距离和在线状态、Yaw/Lift/Claw 当前目标。
- **总线**：CAN2 TX FIFO 空间、BUS-OFF、UART TX 队列；CAN 错误时不能只看上位机“已发送”。

当前 VOFA 绑定仍是预留状态，最可靠的第一阶段调试方法是用调试器观察上述对象和 `Get_Data()` 快照，并同时记录上位机原始 CAN 帧。代码构建和主机回归命令：

```powershell
Set-Location 'E:\gongxun\code\mcu\Control_fixed'
cmake --build build/Integration-Debug --parallel 8
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_host_tests.ps1
```
