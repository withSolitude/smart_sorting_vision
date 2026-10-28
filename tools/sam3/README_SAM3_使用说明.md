# 官方 SAM 3 辅助标注工具

工具目录：`E:\gongxun\27\sam3`。

本工具调用本目录内的官方源码 `sam3-main` 和本地权重 `sam3.pt`，
生成原顶部 YOLO-seg 模型需要的单类别 `object` 实例分割标签。
所有新增 Python 文件都有中文注释，参数集中在 `sam3_config.yaml`。

## 1. 已准备的文件

| 文件 | 用途 |
| --- | --- |
| `sam3_interface.py` | 官方模型接口，支持文字提示、正负框提示、同图共享编码 |
| `sam3_dataset.py` | mask 去重、多边形转换、图片预览、YOLO/CVAT 数据导出 |
| `prelabel_sam3.py` | 递归批量预标注入口 |
| `test_sam3.py` | 单张图片推理和预览 |
| `sam3_config.yaml` | 模型、数据集、提示词、导出参数 |
| `run_test.cmd` | 把图片拖到此文件上，即可测试并打开预览 |
| `run_prelabel.cmd` | 使用配置文件执行批量预标注 |
| `.venv_sam3` | 已创建的运行环境 |

原始 ZIP 和权重保留。官方源码已从 ZIP 解压到 `sam3-main`，
接口通过这个目录导入代码，无须执行官方包的 `pip install -e .`。
模型加载明确设置 `load_from_HF=False`，不会再次申请或下载权重。

## 2. 运行环境

本机已检测到 RTX 4060 Laptop GPU 8 GB，以及能使用 CUDA 的
PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128。

`.venv_sam3` 使用系统已有的 PyTorch、OpenCV 等包，SAM 3 缺少的依赖
装在该虚拟环境内。运行命令必须使用下面的虚拟环境 Python。
从开始菜单直接打开另一个 Python 环境，可能找不到新增依赖。

本接口使用 CUDA＋BF16，保持配置中的 `precision: bfloat16`。
单次只处理一张图片，同图的多个文字提示共享图像编码。

本机已经真实验证本地权重加载、文字提示、正框提示及数据集导出。
堆叠样图上，`red object`、`yellow object`、`white object` 返回的实例分数
分别约为0.98、0.91、0.88；该次推理峰值显存约3.9 GB。
这些数字只是接口运行验证，并不代表整套数据的标注准确率。

若以后重建环境，在已有 CUDA 版 PyTorch 的 Python 环境下运行
`install_sam3.cmd`。本工具不会自动替换或下载 PyTorch。

## 3. 先试一张图片

在 PowerShell 中运行：

```powershell
cd E:\gongxun\27\sam3
.\.venv_sam3\Scripts\python.exe test_sam3.py "E:\smart_sorting_vision\data\datasets\top_seg\v1\val\stacked\top_v1_000379.jpg" --show
```

也可以将一张图片拖到 `run_test.cmd` 上。

结果保存到 `outputs\single_时间戳\`：

- `preview.jpg`：编号、分数和实际导出轮廓；
- `原图片名.txt`：单类别 YOLO-seg 标签。

只想为本次测试更换文字提示时，可这样运行：

```powershell
.\.venv_sam3\Scripts\python.exe test_sam3.py "你的图片.jpg" --prompt "red object" --prompt "yellow object" --show
```

不指定 `--prompt` 时，读取 YAML 中的 `inference.prompts`。
提示结果不等于最终人工标签，需要检查漏检、粘连、把托盘当作物体等情况。

## 4. 数据集路径和目录检查

配置文件已经填写用户提供的路径：

```yaml
dataset:
  source_dir: E:/smart_sorting_vision/data/datasets/top_seg/v1
  splits: [train, val, test]
  output_dir: outputs
  limit: 0
```

目录内部可以保持当前层级：

```text
v1/
├─ train/
│  ├─ single/yellow/圆柱/图片.jpg
│  ├─ stacked/图片.jpg
│  └─ contact/图片.jpg
├─ val/...
└─ test/...
```

先检查路径和图片数量，不加载模型：

```powershell
.\.venv_sam3\Scripts\python.exe prelabel_sam3.py --dry-run
```

`single/stacked/yellow/white` 等只是采集分组，所有模型结果最终统一为
`0: object`。递归读取后保留原来的 train/val/test 划分，不重新随机分组。
如果一个 split 不需要处理，直接从 YAML 的 `splits` 中移除。

## 5. 少量试跑，再处理全部图片

每个 split 先处理前5张：

```powershell
.\.venv_sam3\Scripts\python.exe prelabel_sam3.py --limit 5
```

检查新生成的预览图，调整 YAML 中的英文提示词和置信度。
确认效果后运行全部：

```powershell
.\.venv_sam3\Scripts\python.exe prelabel_sam3.py
```

也可以运行 `run_prelabel.cmd`；命令行试跑为 `run_prelabel.cmd --limit 5`。

每次运行会新建独立的 `outputs\时间戳\`，不会覆盖上一次人工修改过的标签。
源图片只读取和复制，不修改尺寸、文件名或原始字节。

## 6. 输出结果和 CVAT 复核

```text
outputs/时间戳/
├─ dataset/
│  ├─ data.yaml
│  ├─ classes.txt
│  ├─ train.txt
│  ├─ val.txt
│  ├─ test.txt
│  ├─ images/train/原文件名.jpg
│  ├─ images/val/原文件名.jpg
│  ├─ images/test/原文件名.jpg
│  ├─ labels/train/原文件名.txt
│  ├─ labels/val/原文件名.txt
│  └─ labels/test/原文件名.txt
├─ previews/train/原文件名.jpg
├─ previews/val/原文件名.jpg
├─ previews/test/原文件名.jpg
├─ manifest.csv
├─ summary.json
└─ sam3_cvat_project.zip
```

在 CVAT 中创建 Project，通过 Project 的 `Import dataset` 导入
`sam3_cvat_project.zip`，格式选择 **Ultralytics YOLO Segmentation 1.0**。
按 train/val/test subset 检查，逐张修正后再导出 YOLO 分割数据训练。

标签每行是 `0 x1 y1 x2 y2 ...`，坐标已经按原图宽高归一化。
导出的 `data.yaml` 使用本目录相对路径，可作为后续 YOLO 训练输入。

### 必须理解的几种标记

- `NO DETECTION - REVIEW`：模型没有检出，不能直接当成确定的空背景。
- `REVIEW`：某个实例存在断开区域或内孔，普通单条 YOLO polygon 无法无损表示。
  当前只导出最大连通区域的外轮廓，详细情况记录在 `manifest.csv`。
- 相邻同色物体可能被 SAM 合成一个 mask，这种错误仍需要人工拆分。
- 实测颜色提示还会把灰色托盘、边缘绿色机构、螺钉等识别为物体。
  在 CVAT 中删掉这些误标；若某个提示持续误选背景，可在 YAML 中移除或改写该提示。
- 多个提示产生的重复实例，按真实 mask 的 IoU 去重；不会仅因框重叠就删掉目标。

当前标注规则是完整可见物体轮廓，包含可见顶面和侧面。
被遮挡的部分不凭空补齐；两个实际物体分别标为两个实例。
真正的空托盘保留空标签，不新增 background 类。

## 7. 从其他 Python 程序调用接口

在本目录下，用 `.venv_sam3` 运行：

```python
from sam3_interface import Sam3Interface, load_config
from sam3_dataset import process_masks, to_yolo_text

cfg = load_config()
model = Sam3Interface(cfg["model"])

# 支持图片路径、PIL RGB 图片和 OpenCV BGR 数组。
predictions = model.predict("你的图片.jpg", cfg["inference"]["prompts"])
objects = process_masks(predictions, cfg["export"])

# 每个实例提供 mask、score、box、prompt；导出后另有 polygon。
print(len(objects))
```

正负框提示使用原图像素坐标，不需要自己归一化：

```python
model.set_image("你的图片.jpg")
predictions = model.predict_boxes(
    positive_boxes=[[100, 80, 250, 220]],
    negative_boxes=[[300, 100, 420, 250]],
    text="plastic block",
)
```

这些框是示例提示，SAM 3 可能同时返回画面内的相似实例，并不保证只返回框内一个目标。
每次换提示会清理上一组提示，同一图片仍复用缓存编码。

本工具不调用旧的 `last_debug.prepare_top_dataset`；该旧入口依赖
`images.csv/layouts.csv`，与你现在的分层采集目录不是同一套输入格式。
