# 顶部相机 XY 轴与机械 XY 轴不一致：诊断与修复

## 1. 现象

- 物体在图像**右上角**，夹爪实际跑到**左下角**；
- 图像与机械的 X/Y 方向对不上，且误差是整体性的（不是几毫米的标定误差）。

## 2. 先看仓库历史：这不是代码问题，是"回滚只回滚了一半"

同一分支上最近 5 次提交：

| 提交 | 说明 | 对坐标的影响 |
|---|---|---|
| `f8126d65` | 加入边缘 Yaw 中转（x > -40 先退到 -40 再转向） | 无关坐标轴 |
| `41419f8c` | "Align image XY axes" | **`Frame_Rotation_Deg` 0 → 45**，同时新增 `swap_xy` |
| `f87751f5` | "Keep CoreXY workplane orthogonal and swap image axes" | `Frame_Rotation_Deg` 45 → 0，`swap_xy` 保留 |
| `9e542976` | "Default to physical image XY axis swap" | `swap_xy` 默认开启 |
| `1f88035d` | "Rollback coordinate swap and grasp point compensation" | 删除 `swap_xy`，**`Frame_Rotation_Deg` 保持 0** |

也就是说，Codex 先**误判**为"机构相对 CoreXY 旋转了 45°"，把
`Frame_Rotation_Deg` 从 0 改成 45；发现不对之后，它没有直接改回，而是又加了一个
"图像 XY 交换"来补偿，最后整批回滚。

**回滚后的代码，在坐标这一层已经和"当初能用"的状态一致：**

```cpp
// xy_motion_config.h（当前）
constexpr double Frame_Rotation_Deg = 0.0;   // 与可用版本一致
constexpr int Direction_A = 1;
constexpr int Direction_B = 1;
```

```python
# pixel_to_robot.py（当前）—— 没有任何坐标交换
```

那么问题只可能出在**代码之外**：`pixel_to_robot.json`。

> 这个文件在 `data/calibration/...` 下，**不在 git 里**。
> 调试那几天如果在"改过 `Frame_Rotation_Deg` 或开着 `swap_xy`"的代码上重跑过
> 四点标定，错误的轴序就被固化进这个文件了；后来 `git rollback` 能回滚代码，
> 但回滚不了这个 JSON。

## 3. 怎么确认（2 分钟，机器不动）

1. 打开 `data/calibration/top_camera/robot_mapping/pixel_to_robot.points.json`，
   看当初点击的 4 个点的 `robot` 值；
2. 如果 4 个点的 X、Y 量级明显"该是 X 的位置放了 Y"（例如托盘左右两点的 Y 相同、
   X 不同，而实际应该是 X 相同、Y 不同），就是这个文件被写反了。

## 4. 修复

### 方案 A：重跑四点标定（推荐，5 分钟，唯一能根治）

```bash
cd ~/smart_sorting_vision/smart_sorting_vision
python -m scripts.last_debug.calibrate_pixel_to_robot --verify-point
```

加 `--verify-point` 会多要一个第 5 点，用 4 点拟合、第 5 点独立验证：

- 残差 ≤ 5 mm → 通过；
- 残差 > 5 mm → **拒绝保存**，说明 4 点的点击顺序和输入的机械坐标顺序不一致。

同时会打印方向自查表：

```
方向自查（请与实机对照）：
  图像向右 (P1->P2): ΔX= +171.90 mm  ΔY=   +0.00 mm
  图像向下 (P1->P4): ΔX=   +0.00 mm  ΔY= +194.40 mm
```

对照实机：把夹爪往图像右侧移动时 X 是不是真的变大。不一致就重来。

> 4 点单应是**精确解、残差恒为 0**，所以不带第 5 点验证时，点序配错也能"完美保存"——
> 这正是当初没被发现的原因。

### 方案 B：不重标，一键交换轴序

如果现场不方便重新点（托盘已固定、夹爪挡视线），把标定文件当作 `[Y, X]` 顺序，
运行时交换：

```yaml
# config/runtime_config.yaml
calibration:
  pixel_to_robot_swap_xy: true
```

改成 `true` 后**整条链路统一生效**（主程序、调试脚本、标定验证脚本都通过
`camera.top.swap_robot_xy` 读取同一份标志，见 `runtime/app_config.py`）。

验证方式：先跑预览类脚本（把目标点画回图上），确认方向对了再上机械。
改回 `false` 即完全还原。

> 注意 `swap_xy` 是**镜像**变换（X↔Y），不是 180° 旋转。
> 如果现象是"绕中心 180°"（右上 → 左下，且左右上下都反），那更可能是标定文件整体
> 转了 180°，此时 `swap_xy` 修不好，必须重标（方案 A）。

### 方案 C：只想先看看现在映射成什么样

```bash
python -c "
import sys; sys.path.insert(0,'.')
from runtime.app_config import load_runtime_config
from vision.calibration.pixel_to_robot import PixelToRobotTransform
cfg = load_runtime_config('.', 'config/runtime_config.yaml')
t = PixelToRobotTransform.load(cfg['calibration']['pixel_to_robot'], camera_config=cfg['camera']['top'])
print('swap_xy =', t.swap_xy)
for p in [(900,300),(1500,300),(1500,900),(900,900)]:
    print(p, '->', tuple(round(v,2) for v in t.pixel_to_robot(*p)))
"
```

把 4 个角的映射结果和实际位置对一眼就能判断轴向。

## 5. 为什么用配置开关而不是直接改死

- 现场只有一台机器，但"是 [X,Y] 还是 [Y,X]"取决于**标定文件怎么来的**，不是代码能推出来的；
- 开关是**可逆的**：改错了立刻改回来，不需要重新编译、不需要重标；
- 这个开关在 `9e542976` 里已经存在并跑过，本次只是恢复，不是新设计。

## 6. 本次改动

| 文件 | 改动 |
|---|---|
| `vision/calibration/pixel_to_robot.py` | 恢复 `swap_xy`：构造参数、`load()` 推断、`save()` 落盘、正向/反向/批量转换一致 |
| `runtime/app_config.py` | 读取 `calibration.pixel_to_robot_swap_xy`，注入 `camera.top.swap_robot_xy`，供所有入口共用 |
| `config/runtime_config.yaml` | 新增 `pixel_to_robot_swap_xy`（**默认 false**，与当前可用状态一致） |
| `main.py` | 启动时打印实际生效的 `pixel_to_robot_swap_xy`，方便现场确认 |
| `tests/test_pixel_to_robot_axis_swap.py` | 恢复轴序测试，并补一条"默认不交换"的用例 |
| `scripts/last_debug/calibrate_pixel_to_robot.py` | 加 `--verify-point`（第 5 点独立验证）、方向自查表、工作区检查、`det<0` 镜像告警；默认 4 点流程不变 |

## 7. 判错对照表

| 现象 | 结论 | 用哪个 |
|---|---|---|
| 左右、上下**都**反（绕中心 180°） | 标定文件整体转了 180° | 方案 A（重标） |
| X/Y **互换**（图像横向对应机械纵向） | 轴序 [Y,X] | 方案 B（`swap_xy: true`） |
| **只**左右反 | 单轴镜像 | 方案 A |
| 误差几毫米、方向随机 | 标定精度/畸变 | 重标内参，与本问题无关 |
| 误差随离中心距离线性变大 | 尺度错 | 检查 `resize` / ROI 原点 |

## 8. 顺带提醒

`1f88035d` 这次回滚还把**边缘 Yaw 中转**的一部分配置回滚了，但 `edge_motion_config.h`
和 `holding_jaw.cpp` 里的 `Move_XY_Yaw_Via_Edge()` 仍在，边界值 `-40.0` 与上位机
`edge_yaw_limit.x_threshold_mm`（回滚时被删）不再成对存在。若之后要重新启用边缘限角，
需要把 YAML 里的 `edge_yaw_limit` 一并加回来，否则 MCU 会绕行、上位机不会限角。
