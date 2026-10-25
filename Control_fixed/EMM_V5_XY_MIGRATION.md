# Emm V5 CAN / 旋转 45° CoreXY 接入说明

当前工作目录为 `E:\gongxun\code\mcu\Control_fixed`。后续源代码、测试和构建均在此目录进行；复制过来的旧 CMake 缓存不可直接复用。

本工程已将 XY 抓取、托盘移动、识别姿态、回程、独立标定与停止统一接到新的 CAN 运动层。Jetson 的 XY 坐标接口保持不变：上位机仍发机械工作区的 X/Y 毫米坐标，不要在视觉侧重复做 45° 或 CoreXY 变换。

## 1. 驱动器设置与配置入口

集中配置文件：`User_File/3_Robot/xy_motion_config.h`。

| 项目 | 当前配置 | 实机要求 |
| --- | --- | --- |
| 总线 | CAN2，Classic CAN 扩展帧 | CAN2 使用 PB5/PB6，CANH/CANL、共地及总线两端终端电阻正确；CAN1 保留 DM4310 |
| 电机 A/B 地址 | 1 / 2 | ID_Addr 分别设置为 1、2 |
| 波特率 | 1 Mbps | CAN_Baud=1000000；STM32 CAN2、两台 Emm 和 TTL-CAN 必须一致 |
| 通讯功能 | CAN | P_Serial=CAN1_MAP（驱动器菜单名称），并将 TTL-CAN 接到 STM32 CAN2 |
| 校验 | 固定 0x6B | Checksum=0x6B |
| 应答 | 依赖接收确认 | Response=Receive 或 Both；不能选 None/Reached/Other |
| 速度单位 | 整数 RPM | S_Vel_IS=Disable |
| 电机步距/细分 | 1.8°、16 细分，3200 pulse/rev | MotType / MStep 与配置一致 |
| 每转带动长度 A/B | **40 mm，仅占位待标定** | 填实际带轮周长 / 传动比；A/B 可以不同 |
| 机构角度 | +45° | 定义为机构 U 轴相对视觉 +X 逆时针；顺时针改 -45° |
| 电机正方向 | Direction_A/B=+1 | 协议方向 0=CW；装配反向将对应参数改 -1 |
| 最大速度 | 300 RPM | 当前软件发出的运动指令上限，调试先低速 |
| 加速度 | 0 | 恒速，两轴按目标位移比例分配速度；整数 RPM 存在量化误差 |
| 工作区 | X∈[-77,0]，Y∈[0,55] mm | 沿用旧机构数值，新机构必须重新测量 |
| 标定确认 | **Mechanical_Calibration_Confirmed=false** | 机械参数、方向和边界确认后改 true 并重新编译 |

Emm 地址 1 的扩展帧 ID 为 0x0100，广播同步只有 Emm 地址 1 回复，因此 A/B 地址至少有一个必须是 1。当前工程 CAN2 使用 1 Mbps；如果以后选择其他波特率，必须同时修改 FDCAN2 位时序、两台 Emm 和 TTL-CAN 适配器，不能只改上位机下拉框。

## 2. 坐标矩阵与单位

定义视觉 / 工作区坐标为 `(x,y)`，旋转后的 CoreXY 机构坐标为 `(u,v)`，电机带长位移为 `(a,b)`，θ=Frame_Rotation_Deg：

```text
u =  cosθ*x + sinθ*y
v = -sinθ*x + cosθ*y
a = Direction_A * (u+v)
b = Direction_B * (u-v)
```

反馈逆变换先去掉电机正反方向，再计算 `u=(a+b)/2`、`v=(a-b)/2`，最后用逆旋转矩阵恢复 x/y。默认 +45° 时，理想关系是 `a=√2*y`、`b=√2*x`；这个 √2 不能省略。若装配的 A/B 定义不同，须交换物理地址或调整方向，低速逐项确认。

驱动的 0x31、0x36 位置单位为 **65536 tick/rev**，位置指令 0xFD 单位为 **Full_Steps_Per_Rev×Microsteps pulse/rev**。代码分别换算，禁止把两者直接等同。使用 double 做坐标换算，int64_t 保存有符号位置；0xFD 及 0x36 的协议范围仍受 32 位幅值限制，越范围拒绝运动。

原点确认时记录 `Origin_A/B` 和已知 `(Origin_X,Origin_Y)`。默认开启上电置零：初始停止流程结束后，等 CAN2 两台电机的位置、速度、状态反馈新鲜且停稳，将上电时当前位置记录为软件原点；软件不会给驱动器发清零命令来掩盖位置变化。手动 `Confirm_Origin_mm()` 仍可作为调试覆盖入口。

## 3. 单圈与多圈处理

- **0x36 作为运动闭环的位置来源。** 驱动器在持续通电时已累计多圈实际位置，软件解析符号和 32 位幅值，不把目标脉冲数当作实测位置。
- **0x31 独立做软件多圈累计。** `Motor_A/B.Get_Data().Software_Encoder_Ticks` 为相对首次样本的 64 位累计量，支持正反转跨 0/65535；除以 65536 得到累计圈数。位置由累计编码器而非输出脉冲计算。
- 当前每 16 ms 读取一次每台电机的 31/36/35/3A。单圈展开只在连续反馈间隔 ≤80 ms、且每次实际转动小于半圈的假设下有效；300 RPM 的半圈是 100 ms。恰好半圈或超时会锁存 `Encoder_Unwrap_Valid=false`，不猜测缺失圈数；人工重新确认原点后重新采样。软件转速上限不限制手动外力或其他主站指令。
- 31 的软件展开作为独立诊断，不作为 36 故障时的自动替代位置源。单圈诊断丢圈不会静默覆盖闭环位置。
- **断电不能恢复多圈绝对位置。** STM32/驱动器重启后必须在机械零点上电；软件只把当时读取到的当前位置设为新的软件零点，不发送清零命令，也不虚构碰撞回零或断电圈数持久化。

## 4. 启动、停止、故障

上电先发两台驱动的独立立即停止命令，消除 MCU 单独重启时电机继续旧运动的风险。未完成停机确认、未确认原点或标定配置未确认时，START/PICK/CALIB 不能进入运动。

运动使用非阻塞状态机：使能 A → 接收确认 → 使能 B → 接收确认 → 暂存 A 绝对位置 → 接收确认 → 暂存 B 绝对位置 → 接收确认 → 广播同步 → 地址 1 同步确认 → 等待真实位置、转速、到位标志连续稳定 150 ms。每条运动命令的两个 CAN 分包会先检查 TX FIFO 可用空间。运动指令返回 HAL_OK 只表示被运动层接受。

STOP 先给两台电机各发一条 FE，后等接收确认与新位置 / 速度 / 状态反馈停稳；不会因发送成功就回复已停止。驱动禁用时 FE 可能返回 E2，只有新反馈同时证明禁用和静止，才能认定停止。总线 BUSY 会重试，命令确认超时 500 ms、反馈陈旧超过 150 ms、堵转、异常失能、位置跳变或运动超过 15 s 会锁存故障并尝试停止两轴。

`Get_Stop_Confirmed()` 表示实测静止；`Get_Motion_Finished()` 表示目标实际到位。取消目标并停稳不会将原目标标成到位。

故障后原点失效；恢复通信不会自动恢复作业。异常回程不会在 XY 仍未到位时松爪，故障状态保留夹爪；STOP ACK 等真实停止确认。PC1 在作业及 XY/Yaw 标定期间均可按原操作长按 1.5 s STOP。正常 STOP 的机械回程语义沿用原工程。

诊断入口：`XY_Motion.Get_Fault_Code()`、`Motor_A/B.Get_Data()`，包括 Position_Ticks、Software_Encoder_Ticks、Enabled/Reached/Stall、反馈时间戳、应答计数、坏帧计数和展开失效计数。

## 5. 首次调试顺序

1. 设置驱动器菜单，空载完成编码器校准；确认 CAN2 上两台 Emm 均能反馈，CAN1 的 DM4310 保持独立。
2. 测量 A/B 每转带长、确认 A/B 安装和正反方向、45° 的正负、机械工作区；把实际值写入配置。重新检查托盘、识别、待机坐标是否还在新机构可达范围。
3. 确认以上参数后，将 `Mechanical_Calibration_Confirmed` 改 true、重新编译。首次运动用小目标、低速度和空载检查方向与尺度；不能将本文件的 40 mm 占位值视为实测结果。
4. 每次上电前将滑台放到机械零点，等待 `Get_Power_On_Origin_Captured()` 为真。需要覆盖时，再在整机停止、无待执行任务时使用 `XY_Origin_Confirm_Request`；禁止在运行中移动原点。
5. 测试小幅单方向 XY 标定，比较实际移动量、恢复的 XY 坐标和原点，再测试跨多圈及反向回零。确认不会重复应用视觉侧变换。
6. 测试运动中 STOP、CAN 断线、禁止一台电机使能、重新上电；确认不会报告虚假到位 / STOP完成，恢复后要求重新确认原点。
7. 最后验收目标抓取 → 托盘 → 识别姿态 → 分箱 → 回待机全流程。

## 6. 编译与回归

```powershell
Set-Location 'E:\gongxun\code\mcu\Control_fixed'
cmake -S . -B build/Integration-Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Integration-Debug --parallel 8
powershell -ExecutionPolicy Bypass -File tests/run_host_tests.ps1
```

产物位于 `build/Integration-Debug/Control.elf`、`Control.hex`、`Control.bin`。2026-10-24 已完成新路径下 ARM GCC 13.3.1 编译链接，32 组主机回归全部通过：11 组驱动/XY、1 组默认运动锁、11 组真实夹爪/储物状态机、6 组真实握手状态机和 3 组整机联合调度。构建仍有工程原有的 CMSIS / 通用算法警告，主机回归使用 `-Wall -Wextra -Werror` 编译通过。测试覆盖说明见 `tests/README.md`。

测试使用独立 HAL/CAN 模拟器，测试宏不会打开实机配置；固件不包含测试代码。本次为软件编译和模拟回归，尚未连接实际机构验证方向、行程、回零或供电复位行为。

协议依据：用户提供的《Emm42_V5.0 步进闭环驱动说明书 Rev1.3》及[张大头厂商原文](https://blog.csdn.net/zhangdatou666/article/details/132644047)。

## 7. 上层接口与调度整合

`holding_jaw` 的全部 XY 状态通过 `XY_Motion.Move_To_mm()` 提交毫米目标，并按当前阶段预期目标坐标核对实测反馈。`HAL_OK` 表示目标被接受，`HAL_BUSY` 保留当前步骤重试，禁止拿旧运动的完成标志推进新抓取。

| 模块接口 | 当前含义 |
| --- | --- |
| `Holding_Jaw.Get_Disable_Ready()` | 真实 XY 在待机坐标、原点有效、全部舵机命令时间已结束，且没有独立 Yaw 标定覆盖 |
| `Holding_Jaw.Get_Stop_Ready()` | XY 实测停稳，舵机完成正常待机或故障停机命令；不要求断电后尚未建立的原点 |
| `Holding_Jaw.Get_Outputs_Settled()` | 当前状态最后一条舵机指令发送成功后已等待设定动作时间；这不是舵机编码器反馈 |
| `Holding_Jaw.Control_Calibration_Yaw()` / `Stop_Calibration_Yaw()` | 统一记录独立 Yaw 标定和停机，标定会使真实待机条件失效，直到显式 Return 完成 |
| `Sorting_Storage.Get_Disable_Ready()` | 转盘位置、速度和使能反馈新鲜，位于待机方向且低速，托盘复位指令时间完成 |
| `Sorting_Storage.Get_Fault()` / `Clear_Fault()` | 故障锁存；只有物理待机时才可清除，自动分拣不能自行清锁重启 |

Return 先抬升 Lift 并等待原有 500 ms 动作时间，再移动 XY/Yaw，实测到达待机坐标后才张爪。所有活动状态包含输出 BUSY 等待都有 30 秒阶段超时。固定托盘、识别和待机位置、速度及舵机角会在抓取开始前检查，避免夹住物体后才发现后续目标非法。原有机械标定值没有自动重算。

正式 START 若发现标定后实际位置不在待机，会先回待机，再通知 Jetson 开始工作。`RETRY_PICK_READY`、`SORT_DONE` 均等待夹爪真实待机；`SORT_DONE` 还需储物机构完整投放并复位。完成通知入队 BUSY 时只重试通知，不重做机械动作。独立储物测试与 XY/Yaw 标定互斥，XY/Yaw 启动遇短暂 BUSY 保留待执行请求。

Storage 仍使用 DM4310 和翻转舵机。新增普通状态/速度反馈计数与时间戳，区分普通反馈与 0x51 输出轴位置读回应答；两者任何一路超过 100 ms 未更新都会禁止继续闭环运动并锁存故障。到位稳定计数只消费新的输出轴位置样本。方向为 -1 时位置、速度及最终力矩一起换算。

故障上报只建立一次停止事务，错误帧遇 TX 队列 BUSY 会保留并重试。显式 STOP 有独立来源标志，同周期出现故障不会抹掉人工恢复请求。Storage 故障后需 PC1 再次长按 1.5 s 或 Jetson 再发 STOP，才请求恢复；物理待机后清锁，不自动 START。XY 故障停稳可确认 STOP，但继续作业仍需重新确认原点；STOP ACK 与 START Ready 不再混为一个条件。故障状态下 PC1 短按不会启动，长按可发出恢复 STOP。

CAN/UART 回调统一经过机构接口路由；1 ms XY 位置更新和待停机服务在 10/100 ms 上层状态机之前执行。本次修改前的工程备份在 `E:\gongxun\code\mcu\Control_fixed_before_upper_integration_20261024.zip`。
