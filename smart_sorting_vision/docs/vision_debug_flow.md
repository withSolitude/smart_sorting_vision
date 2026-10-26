# 视觉系统从零调试流程

本文用于第一次接入顶部相机、侧面相机和 STM32 后的现场调试。所有长期参数只写入
`config/runtime_config.yaml`；`scripts/last_debug` 只负责读取配置、显示画面和保存数据，
不要再在脚本里复制一份相机分辨率、曝光、ROI 或模型路径。

工作目录：

```bash
cd <工程目录>/smart_sorting_vision
export PYTHONPATH=.
```

Windows 调试机可使用：

```powershell
cd E:\gongxun\code\smart_sorting_vision\smart_sorting_vision
$env:PYTHONPATH = (Get-Location).Path
```

## 1. 上电前检查

1. 给相机、Jetson、STM32 和 USB-UART 供电，确认机械机构处在安全待机位置。
2. 在 Jetson 上确认设备名：

   ```bash
   ls -l /dev/v4l/by-id/
   v4l2-ctl --list-devices
   ls -l /dev/ttyCH341USB0
   ```

3. 把实际设备名填入 `camera.top.device`、`camera.side.device` 和 `stm32.device`。
4. 检查 `model.*.path`、`calibration.*` 的文件存在；不要为了让程序启动而把路径改成脚本中的临时路径。
5. 每次调试前复制一份 YAML，例如 `runtime_config.before_debug.yaml`，便于回退。

## 2. 顶部相机画面、曝光和 ROI

先只调画面，不加载模型：

```bash
   python -m scripts.last_debug.preview_cameras --camera top --raw
```

`--raw` 用于内参文件尚未生成时的原始画面预览；完成内参标定后，去掉 `--raw` 检查
去畸变画面。生产程序使用去畸变后的画面。

按窗口提示退出。脚本读取 YAML 的分辨率、帧率、FourCC 和 `controls`，画面稳定后再调：

- 曝光：物体边缘不能过曝，黑色区域不能完全丢失；
- 增益：优先保持较低，避免噪声被分割模型当成物体；
- 白平衡：固定后再采集训练数据；
- 帧率和缓存：保证画面连续，不出现明显延迟。

参数只改 `camera.top.controls`，改完重新运行预览确认。然后用同一个预览脚本显示 `safe_roi`：

```bash
   python -m scripts.last_debug.preview_cameras --camera top
```

ROI 应覆盖完整工作区，同时排除支架、边框和无关背景。托盘边缘有圆弧或不是标准矩形时，运行多边形标注工具：

```bash
   python -m scripts.last_debug.select_roi --camera top --write
```

窗口会持续显示实际相机画面。鼠标左键逐点点击托盘边界，沿圆弧多点取样；按 `R` 清空重画，
按 Backspace 删除最后一点，按 Enter 确认，按 `Q` 或 Esc 取消。顶部有内参时显示去畸变画面，
没有内参时显示原始画面。确认后工具会同时写入：

```yaml
camera:
  top:
    safe_roi: [x1, y1, x2, y2]             # 多边形外接矩形，保持原输出尺寸
    safe_roi_polygon: [[x1, y1], [x2, y2]]  # 实际手工边界，可包含任意数量的点
```

正式程序仍返回原来的矩形尺寸，只将多边形外部像素置零，所以像素到机械坐标的尺寸和接口不变。
如果这一步是在内参文件生成前完成的，先把它当作临时边界；完成顶部内参标定后，重新运行同一命令，
在去畸变画面上再点一遍并覆盖保存最终边界。
不要在顶部处理器里再写第二个 ROI。

## 3. 侧面相机画面、曝光和 ROI

```bash
python -m scripts.last_debug.preview_cameras --camera side
```

侧面 ROI 必须覆盖识别托盘和物体可能出现的区域，排除传送结构和固定背景。侧面也使用同样的实时多边形点选：

```bash
python -m scripts.last_debug.select_roi --camera side --write
```

保存后 YAML 中会同时有 `camera.side.roi` 外接矩形和 `camera.side.roi_polygon` 点集。多边形外的像素会置零，
因此修改侧面多边形后必须重新拍空托盘背景。背景文件由 `calibration.side_background`
指定，采集前托盘上不能有物体、手或工具：

```bash
python -m scripts.last_debug.capture_side_background
```

如果现场光线、托盘位置或相机高度发生变化，应重新拍背景，不要继续使用旧背景图。

## 4. 顶部相机内参标定

使用 6×9 棋盘格，格距必须按实际打印件实测后填写 `calibration.top_camera.square_size_mm`，不能直接照抄示例值。棋盘在画面中覆盖不同位置、距离和角度，至少采集
15 张清晰图片，避免所有图片都在中心位置：

```bash
python -m scripts.last_debug.capture_top_intrinsics
python -m scripts.last_debug.calibrate_top_intrinsics
```

按脚本提示移动棋盘，检测不到角点的图片不要保存。完成后确认输出文件为：

`data/calibration/top_camera/top_camera_intrinsics.npz`

然后用顶部预览脚本检查去畸变后的直线和 `safe_roi` 是否仍覆盖工作区。

## 5. 顶部像素到 XY 坐标标定

1. 下位机上电，确认 CAN2 两台 EMM 反馈正常；按当前工程设定，当前机械位置先作为软件原点。
2. 上位机只发送 `STOP`、心跳和独立 `CALIB_MOVE`，不要直接启动正式分拣。
3. 在工作区选取至少 5 个不共线点，记录相机像素中心和下位机返回的 XY 坐标。
4. 使用四个固定点拟合映射，检查每个点的残差：

   ```bash
   python -m scripts.last_debug.calibrate_pixel_to_robot
   ```

   边缘点残差明显变大时，重新检查内参、ROI 和机械坐标方向。
5. 将最终映射写入 `calibration.pixel_to_robot` 指向的 JSON。视觉代码只读取这个文件，不能在处理器中再写矩阵常量。

验证要求：中心点和四个边缘点都能落在 `mechanics.workspace_mm` 内；X/Y 正方向、单位和 STM32 返回值一致。

## 6. 顶部物体数据和抓取点

确认画面、内参和像素到 XY 映射后再采数据。数据应覆盖比赛现场的颜色、形状、大小、位置、遮挡和
反光情况，顶部和侧面数据分开保存：

```bash
python -m scripts.last_debug.capture_frames --camera top
python -m scripts.last_debug.capture_frames --camera side
```

数据采集时固定相机参数和 ROI；如果修改曝光、白平衡或 ROI，应重新采集受影响的数据。训练完成后只在
YAML 修改 `model.top.path`、`model.top.conf`、`model.top.imgsz`，再运行顶部单帧/连续推理：

```bash
python scripts/debug/test_top_pick_pipeline.py
```

先观察分割轮廓和中心点，再观察抓取轴方向、预开口量和最终 XY。不要在正式运行前同时调模型阈值、ROI、
坐标矩阵和机械补偿；一次只改一类参数。

## 7. 侧面识别调试

侧面流程先把完整机械 ROI 送入形状分类 YOLO。模型必须包含 `no_object`
空背景类别；只有模型确认是真实形状后，才进入颜色和二维码识别。背景差分仅
作为已有物体的裁剪辅助，不能再决定“有没有物体”：

```bash
python scripts/debug/test_side_sort_pipeline.py
```

调试顺序：

1. 在 `data/datasets/side_cls/v1/{train,val,test}/no_object/` 保存不同环境下的空托盘完整 ROI；
2. 同样保存每种形状的完整 ROI，再训练侧面分类模型；
3. 把模型类别名 `no_object`（或配置中的其它空背景名）写入 `no_object_labels`；
4. 空托盘测试应直接输出 `no_object`，跳过颜色/二维码，发送 RETRY_PICK 回到 Disable 后重新开始顶部识别；
5. 有物体时分别测试每种颜色、有/无二维码、反光二维码和每种形状；
6. 最后再调 `vision.side.consensus` 的窗口和通过帧数。

只检查形状和空背景时可运行：

```bash
python scripts/debug/test_side_shape.py
```

颜色、二维码和形状的阈值只写在 YAML 或其明确引用的模型配置中；不要在多个分类脚本中复制同一阈值。

### 二维码识别模式

`config/runtime_config.yaml` 的 `vision.side.recognition.qr` 有两个独立开关：

- `enabled: false`：完全关闭二维码识别；
- `enabled: true` 且 `check_content_consistency: false`：只确认有无二维码；
- `enabled: true` 且 `check_content_consistency: true`：除确认存在外，还要求共识窗口内的解码内容一致。

内容校验只影响多帧 READY 共识，不改变二维码检测器的定位和解码流程；比较时会去掉解码文本首尾空白。

## 8. 上下位机通信联调

先不接物体，按以下顺序测试：

1. 启动下位机，确认 CAN2 两台 EMM 有反馈，XY 原点有效，夹爪和储物机构在待机点。
2. 使用通信调试脚本发送心跳，确认持续收到 `MCU_HEARTBEAT_ACK`。
3. 发送 `STOP`，确认收到安全 `MCU_STOP_ACK`；收到 ACK 前不要发送抓取命令。
4. 用 `CALIB_MOVE` 做小范围单点移动，确认协议坐标、视觉坐标和机械实际方向一致。
5. 用一个物体做完整链路：`PICK_PREOPEN → PICK_YAW_BRANCH → PICK_TARGET → PICK_ACK → RECOGNITION_READY → SORT_RESULT → SORT_DONE`。
6. 测试拔插 USB-UART 或暂时停止下位机：上位机应进入 `LINK_OFFLINE`，停止当前周期并重连；恢复后重新 STOP 同步，不应退出进程。

通信测试中只改变一个变量。若收到 `MCU_ERROR`，先查看下位机机械反馈和错误码，不要用提高超时阈值掩盖真实故障。

## 9. 最终验收顺序

```text
相机设备名
→ 画面与曝光
→ 顶部/侧面 ROI
→ 顶部内参
→ 像素到 XY 映射
→ 空背景
→ 顶部数据/模型
→ 侧面数据/模型
→ 心跳与 STOP
→ 单点 CALIB_MOVE
→ 单物体完整分拣
→ 连续多物体分拣
```

每一步完成后保存 YAML、标定文件、背景图和测试日志。只有单物体完整链路稳定后，才进入连续分拣测试。
