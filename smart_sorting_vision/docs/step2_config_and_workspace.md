# Step 2 — 正式配置与机械工作区

## 已固定的机械工作范围

- X：`-77.0 ~ 0.0 mm`
- Y：`0.0 ~ 55.0 mm`

用户描述的 X 运动方向为“0 到 -77”；配置文件统一按 `[min, max]` 表示，因此写作 `[-77.0, 0.0]`。

## 本步骤修改

1. `runtime_config.yaml` 成为正式运行的唯一配置入口。
2. 模型、标定文件改为相对工程根路径，不再依赖 `/home/nvidia/...`。
3. 顶部/侧面相机设备、分辨率、ROI、曝光、白平衡等进入 YAML。
4. TopCamera / SideCamera 在启动时从 YAML 注入 profile。
5. TopPickPipeline 使用真实 XY 工作范围过滤候选抓取点。
6. SortingRuntime 在发送 `PICK_TARGET` 前再次检查 XY，形成双重限位。
7. 顶部正式投票固定为 2/2，侧面正式投票固定为 4/5，并在启动日志打印。
8. 新增 `python -m scripts.check_runtime_config` 做启动前检查。
9. 新增 `python -m scripts.test_step2_config_and_limits` 做离线配置/限位回归。

## Jetson 上下一步

在工程根目录执行：

```bash
cd ~/smart_sorting_vision
source .venv/bin/activate

ls -l /dev/v4l/by-id/
python -m scripts.check_runtime_config
python -m scripts.test_step2_config_and_limits
```

如果 `check_runtime_config` 中 top/side 设备显示 `MISSING`，只修改：

`config/runtime_config.yaml -> camera.top.device / camera.side.device`

不要再去改 `top_camera.py` 或 `side_camera.py` 的 DEVICE。

设备映射确认后，再进行 Step 3：顶部相机/机械坐标重新标定与侧面背景重新采集。
