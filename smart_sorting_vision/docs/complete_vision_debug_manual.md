# 智能分拣视觉系统完整调试手册

本文对应分支 `codex/robust-reconnect-cleanup`，覆盖相机参数、顶部内参、ROI、像素到机械 XY、侧面背景、数据集、模型训练、模型部署和整机验证。

所有长期参数只写入：

```text
config/runtime_config.yaml
```

不要在脚本中复制曝光、ROI、分辨率、模型路径或标定参数。

## 0. 当前视觉流程

### 顶部相机

```text
TopCamera 原始帧
→ 内参去畸变
→ safe_roi 矩形裁剪
→ safe_roi_polygon 多边形掩膜
→ 顶部 YOLO 分割
→ 抓取点、Yaw、夹爪开口
→ 像素到机械 XY
```

### 侧面相机

```text
SideCamera 原始帧
→ roi 矩形裁剪
→ roi_polygon 多边形掩膜
→ 完整侧面 ROI 形状/空背景分类
→ no_object：跳过颜色和二维码，返回 Disable 后重新抓取
→ 有物体：背景辅助裁剪
→ 颜色、二维码、形状共识
→ 分箱
```

侧面背景差分只负责已有物体的颜色/二维码裁剪，不能再作为“有没有物体”的判断依据。

## 1. Jetson 环境和分支

```bash
cd ~/smart_sorting_vision
git checkout codex/robust-reconnect-cleanup
git pull
export PYTHONPATH=.
```

确认配置能被加载：

```bash
python -c "from runtime.app_config import load_runtime_config; from pathlib import Path; load_runtime_config(Path('.').resolve(), Path('config/runtime_config.yaml')); print('config ok')"
```

确认相机设备：

```bash
ls -l /dev/v4l/by-id/
v4l2-ctl --list-devices
```

确认设备、STM32 和相机供电正常。调试时先让机构处在安全待机位置，暂停正式分拣。

## 2. 当前相机参数

顶部相机当前参数：

```yaml
camera:
  top:
    width: 1920
    height: 1080
    safe_roi: [556, 35, 1633, 975]
    controls:
      auto_exposure: 1
      exposure_time_absolute: 105
      gain: 0
      white_balance_automatic: 0
      white_balance_temperature: 5400
      contrast: 30
      saturation: 55
      sharpness: 3
```

侧面相机当前参数：

```yaml
camera:
  side:
    width: 1280
    height: 720
    roi: [0, 0, 1280, 720]
    controls:
      auto_exposure: 1
      exposure_time_absolute: 85
      gain: 0
      white_balance_automatic: 0
      white_balance_temperature: 4650
      brightness: -3
      contrast: 30
      saturation: 55
      sharpness: 1
```

`auto_exposure: 1` 是 V4L2 手动模式，表示关闭自动曝光；`3` 才是光圈优先自动曝光模式。

调参时一次只改一个参数。每次改曝光、白平衡、分辨率、ROI 或镜头位置，都要重新保存调试照片并记录 YAML。

## 3. 相机画面和参数调试

同时预览顶部和侧面：

```bash
python -m scripts.last_debug.preview_cameras --camera both
```

顶部没有可用内参时查看原始画面：

```bash
python -m scripts.last_debug.preview_cameras --camera top --raw
```

`--raw` 只用于顶部内参标定前的原始画面检查。正式运行和最终 ROI 都使用不带 `--raw` 的去畸变画面。

预览窗口按 `S`，会保存完整图、ROI 标注图、ROI 裁剪图和同名 JSON。默认目录：

```text
data/debug/last_debug/frames/
```

JSON 会记录相机名称、时间、图像尺寸、ROI 外接矩形、ROI 多边形、曝光、增益、白平衡、对比度和锐度。

建议调节顺序：

```text
分辨率和帧率
→ 自动曝光/自动白平衡关闭
→ 曝光时间
→ 增益
→ 白平衡
→ 对比度和饱和度
→ 锐度
→ ROI
```

画面目标：白色/黄色物体不大面积纯白，黑色物体仍有轮廓，灰色托盘稳定，颜色之间有明显差异，画面没有拖影和自动亮度跳变。

## 4. 顶部圆点板内参标定

当前标定参数：

```yaml
calibration:
  top_camera:
    pattern_type: symmetric_circles
    pattern_size: [7, 7]
    circle_spacing_mm: 7.0
    fix_k3: true
    new_camera_alpha: 0.0
    min_valid_images: 12
    warning_error_px: 0.8
    max_view_error_px: 2.0
```

`circle_spacing_mm` 是相邻圆心距离，不是圆点直径，必须按厂家规格或实际测量值确认。

### 4.1 清理旧图片

标定目录：

```text
data/calibration/top_camera/images/
```

只保留顶部相机实际拍摄的图片，不要混入手机图片、侧面相机图片、旧棋盘格图片、不同分辨率图片或旧去畸变图片。正式图片必须是 `1920×1080`。

### 4.2 采集原始标定图片

```bash
python -m scripts.last_debug.capture_top_intrinsics
```

启动后必须看到：

```text
[RAW] 当前采集使用原始 read_full()，没有经过旧内参去畸变。
```

窗口显示 `circles=True` 后按 `S` 或空格保存。建议采集 20～30 张，覆盖中心、四角、不同距离、不同倾斜角和不同旋转角。49 个圆点必须完整可见，避免遮挡、反光和运动模糊。

### 4.3 计算内参

```bash
python -m scripts.last_debug.calibrate_top_intrinsics
```

合格输出应满足：

```text
flags=128
alpha=0
RMS <= 0.8px
单张 view_rmse <= 2.0px
valid >= 12
image_size = 1920 x 1080
```

程序会拒绝分辨率不一致的图片、单张重投影误差过大的图片和整体 RMS 超限的结果，不会保存坏的内参文件。

输出文件：

```text
data/calibration/top_camera/top_camera_intrinsics.npz
```

### 4.4 检查去畸变

```bash
python -m scripts.last_debug.preview_cameras --camera top
```

不要加 `--raw`。检查托盘边缘是否接近直线、是否出现大面积黑色、是否变成圆环、物体是否被拉伸。如果异常，不要继续点 ROI，先重新检查标定图片和 RMS。

## 5. 顶部 ROI

去畸变画面正常后执行：

```bash
python -m scripts.last_debug.select_roi --camera top --write
```

点选坐标是去畸变完整图坐标，不是 RAW 坐标，也不是旧 ROI 局部坐标。沿托盘实际边界点选，左下圆弧和右下机械开口处多点采样。

按 `S` 保存点选快照但不写 YAML；按 `Enter` 写入 YAML。程序会自动计算 `safe_roi` 外接矩形和 `safe_roi_polygon` 多边形。

当前顶部 ROI：

```yaml
safe_roi: [556, 35, 1633, 975]
```

多边形点必须满足：

```text
x1 <= x < x2
y1 <= y < y2
```

修改顶部 ROI 后，必须重新进行像素到机械 XY 标定。

## 6. 顶部像素到机械 XY

```bash
python -m scripts.last_debug.calibrate_pixel_to_robot
```

脚本使用正式 `TopCamera.read()` 输出，也就是去畸变、ROI 裁剪和多边形掩膜之后的图像。依次点击四个固定机械点：

```text
P1 左上
P2 右上
P3 右下
P4 左下
```

四点应覆盖整个有效工作区，不能集中在中间。每个点输入对应的机械 `X_mm Y_mm`。

输出：

```text
data/calibration/top_camera/robot_mapping/pixel_to_robot.json
data/calibration/top_camera/robot_mapping/pixel_to_robot.points.json
```

修改相机位置、分辨率、去畸变参数、safe_roi 或 safe_roi_polygon 后，都必须重新标定这个文件。

## 7. 侧面 ROI 和空背景

侧面 ROI：

```bash
python -m scripts.last_debug.select_roi --camera side --write
```

多边形要包含物体完整可能出现的区域，排除左侧电机、右侧导轨和托盘外机械结构。

侧面 ROI 或相机参数改变后必须重新采集空背景：

```bash
python -m scripts.last_debug.capture_side_background
```

采集背景时托盘必须为空，没有手、工具和标定板，光照要与正式运行一致。

输出文件：

```text
data/calibration/side_camera/side_mechanical_roi_background.png
```

## 8. 顶部实例分割数据集

顶部训练图片必须来自正式 `TopCamera.read()` 坐标系，不能使用 RAW 图片、旧 ROI 图片、手机图片或不同尺寸图片。

数据要覆盖：

- 每种颜色和形状；
- 单物体、多物体、接触、遮挡和堆叠；
- 靠近左、右、上、下边缘；
- 空托盘；
- 不同角度、排列、阴影和反光。

顶部数据目录通常为：

```text
data/datasets/top_seg/raw/
```

采集脚本：

```bash
python -m scripts.dataset.top_seg.capture_top_dataset
```

每个 layout 记录当前托盘中的所有物体。保存的图片再进行人工实例轮廓标注，标注每个物体的轮廓，不要把整张托盘标成一个目标。

如果脚本使用的相机 profile 与 YAML 不一致，先确认脚本已经调用 `TopCamera.configure()`，否则不要用它采集正式数据。

## 9. 顶部数据集整理和检查

建议划分：

```text
train：70%～80%
val：10%～20%
test：10%～20%
```

同一场景的连续帧不能同时进入 train 和 val/test。

验证数据集：

```bash
python -m scripts.dataset.top_seg.validate_top_seg_dataset
```

检查：

- 图片和标签一一对应；
- 标签坐标在 [0,1]；
- 没有空标签；
- 类别名称一致；
- train/val/test 没有重复图片；
- `data.yaml` 路径正确。

## 10. 顶部分割模型训练

准备好 V3 数据后，从已验证的 V2 模型继续训练：

```bash
python -m scripts.train.top_seg.train_top_seg_v3 \
  --data data/datasets/top_seg/prepared/v3/data.yaml \
  --model weights/top_seg/v2/best.pt \
  --name v3_interim_aug \
  --device 0
```

训练前确认 GPU、初始化模型和 `data.yaml` 都存在。训练后检查 box/mask 指标、边缘物体、接触物体、空托盘和各颜色/形状的检出效果。

结果通常在：

```text
outputs/training/top_seg/v3/<run_name>/weights/best.pt
```

## 11. 侧面形状/空背景数据集

侧面形状模型直接看完整侧面 ROI，不保存背景差分 crop。

目录结构：

```text
data/datasets/side_cls/v1/
├── train/
│   ├── no_object/
│   ├── cube/
│   ├── cuboid/
│   └── ...
├── val/
│   ├── no_object/
│   ├── cube/
│   └── ...
└── test/
    ├── no_object/
    ├── cube/
    └── ...
```

采集：

```bash
python -m scripts.dataset.side_cls.capture_side_shape_dataset
```

操作：选择 split，输入类别目录名，空托盘输入 `no_object`，按 `S` 保存完整 ROI。

`no_object` 必须包含不同空托盘位置、不同光照和轻微曝光变化。物体类别要覆盖不同形状、位置、角度、颜色、二维码状态、反光和阴影。

## 12. 侧面分类模型训练

```bash
python -m scripts.train.side_cls.train_side_cls
```

训练前确认 train、val、test 都有 `no_object`。训练后检查：

- 空托盘稳定输出 `no_object`；
- 有物体不会被判成 `no_object`；
- 低置信度样本不会稳定输出错误形状；
- 物体位于 ROI 不同位置时仍可识别。

配置中的：

```yaml
vision:
  side:
    recognition:
      shape:
        no_object_labels: [no_object, empty, background]
```

必须与训练目录名一致。

## 13. 颜色和二维码调试

侧面形状分类确认有物体后才运行颜色和二维码。

```yaml
vision:
  side:
    recognition:
      qr:
        enabled: true
        check_content_consistency: false
```

三种模式：

```text
enabled=false
→ 完全关闭二维码识别

enabled=true, check_content_consistency=false
→ 只判断有无二维码

enabled=true, check_content_consistency=true
→ 还要求多帧解码内容一致
```

二维码至少测试无二维码、正面、旋转、斜视、反光和多帧内容变化。颜色至少测试白、黑、灰、红、橙、黄、绿、青、蓝、紫、粉以及阴影和反光。

## 14. 模型路径和 TensorRT

训练出的 `best.pt` 不能直接当作正式 TensorRT engine。正式 YAML 必须指向实际存在的 `.engine` 文件：

```yaml
model:
  top:
    path: <当前顶部 best.engine>
  side:
    path: <当前侧面 best.engine>
```

侧面 engine 必须来自包含 `no_object` 类别的完整 ROI 分类模型。TensorRT engine 建议在最终 Jetson 上生成，并确认输入尺寸、CUDA/TensorRT 版本和模型版本一致。

## 15. 单模块验证

### 顶部

确认画面、ROI、分割轮廓、抓取中心、Yaw、预开口和边界限制。

### 侧面空托盘

必须出现：

```text
ShapeClassifier → no_object
→ 不运行颜色/二维码
→ RETRY_PICK
→ 下位机回 Disable
→ 新的顶部识别周期
```

### 侧面有物体

必须出现：

```text
ShapeClassifier → 真实形状
→ 颜色
→ QR
→ 4/5 共识
→ READY
```

### 单物体完整流程

```text
顶部识别
→ PICK_YAW_BRANCH
→ PICK_TARGET
→ PICK_ACK
→ RECOGNITION_READY
→ 侧面识别
→ SORT_RESULT
→ SORT_ACK
→ SORT_DONE
```

## 16. 通信和掉线测试

正式运行前验证：

1. 心跳 ACK 持续返回；
2. STOP 能收到 STOP_ACK；
3. PICK ACK 丢失时会重发；
4. READY 丢失时会重发；
5. SORT ACK/DONE 丢失时不会重复扣库存；
6. USB-UART 暂时断开时进入 `LINK_OFFLINE`；
7. 链路恢复后重新 STOP 同步，再继续新周期；
8. 视觉识别超时不会永久锁死为 ERROR。

## 17. 最终验收顺序

```text
相机设备
→ 曝光、白平衡、分辨率
→ 顶部 RAW 画面
→ 顶部圆点板内参
→ 顶部去畸变
→ 顶部 safe_roi 和 polygon
→ 顶部 pixel_to_robot
→ 侧面 roi 和 polygon
→ 侧面空背景
→ 顶部数据采集
→ 顶部分割模型训练
→ 侧面 no_object/形状数据采集
→ 侧面分类模型训练
→ 颜色和二维码调试
→ TensorRT engine 部署
→ 单物体完整流程
→ 空托盘 RETRY_PICK 流程
→ 连续多物体分拣
→ 掉线恢复测试
```

每次正式运行前保存：

```text
runtime_config.yaml
top_camera_intrinsics.npz
pixel_to_robot.json
side_mechanical_roi_background.png
顶部/侧面模型文件
调试照片和对应 JSON
```

出现异常时先记录当前 YAML、模型路径、标定文件、照片和日志，再只修改一个参数重新验证。
