# 侧面相机：按具体形状预标注 YOLO 实例分割

本工具使用本地 SAM 3 生成轮廓，输出供侧面 YOLO-seg 训练的数据集。
标的是整个物体的可见区域，包括顶面和侧面；不标阴影、托盘和不可见部分。

## 1. 类别与数据来源

侧面独立配置为 `sam3_side_config.yaml`，原来的顶部配置仍是 `sam3_config.yaml`。

已经按实际 `side_cls` 目录配置了 12 类：三棱柱、三棱锥、五棱柱、六棱柱、四棱台、
四棱锥、圆台、圆柱、圆锥、正方体、球、长方体。没有合并成“棱柱”“棱锥”大类。

默认输入 `E:\smart_sorting_vision\data\datasets\side_cls\school_mixed`，检查时共有
train 854 张、val 203 张、test 0 张，覆盖完整 12 类。

目录格式如下，颜色只作为子目录，不增加 YOLO 类别：

```text
school_mixed/
├─ train/
│  ├─ 三棱柱/blue/图片.jpg
│  ├─ 五棱柱/yellow/图片.jpg
│  └─ ...
├─ val/
└─ test/
```

`annotation.classes` 的列表顺序就是类别 ID，从 0 开始。
开始标注或训练后不要调换顺序；加新形状时在末尾增加一项。

**细分类别来自形状目录，SAM 3 负责找轮廓。** 这能避免仅靠文字提示猜测五棱柱、六棱柱。
如果原图放错目录，应纠正原目录，或在交互窗口改成实际类别。
单幅图看不出底面边数时，也不能把 SAM 的分数当作形状判定证据。

## 2. 推荐：交互式标注

直接双击 `run_side_interactive.cmd`。

1. 等待本地模型加载，窗口按当前图片所在目录预选形状。
2. 左键点物体内部；右键点误选的背景。
3. 这里需要整个可见物体。顶面、侧面都应保留，必要时分别补正点。
4. 用 `1 / 2 / 3` 比较同一个物体的候选轮廓。
5. 检查右侧类别，例如五棱柱，按 Enter 确认实例。
6. 一张图有多个目标时，为每个目标选择实际类别并分别确认。
7. 按 S 保存；N 保存并下一张；E 导出已保存图片的 CVAT 包。

分割轮廓不对时，按 P 手工描边，至少三个点，Enter 闭合并确认。
侧面物体的圆弧可以多点描边。滚轮缩放、中键拖动不改变保存的坐标。

修改已确认实例的类别：在实例列表选中它，在类别下拉框选择正确形状，再点
“将当前类别应用到选中实例”，最后 S 保存。仅切换下拉框不会改动已确认标签。
类别下拉框有焦点时，键盘交给下拉框操作；要用标注快捷键，先点击图像区域。

交互模式不使用 `prompts`，所以不受形状文字提示是否能检出的限制。
再次启动同一输出目录会继续未保存的图片，也能返回修改已经保存的图片。
详细点选、框选和快捷键操作见 `README_交互式标注.md`。

默认输出：

```text
outputs_side_interactive/
├─ annotations.json           # 像素轮廓、类别和恢复记录
├─ dataset/
│  ├─ data.yaml               # 12个具体类别，可用于YOLO-seg训练
│  ├─ classes.txt
│  ├─ train.txt / val.txt / test.txt
│  ├─ images/train/图片__路径后缀.jpg
│  └─ labels/train/图片__路径后缀.txt
└─ sam3_interactive_cvat.zip   # 按E生成或更新
```

## 3. 可选：先自动预标注，再到 CVAT 复核

先检查目录与类别，不加载 GPU：

```powershell
cd E:\gongxun\27\sam3
.\run_side_prelabel.cmd --dry-run
```

先跑每个 split 的前 5 张：

```powershell
.\run_side_prelabel.cmd --limit 5
```

确认效果后，全量运行：

```powershell
.\run_side_prelabel.cmd
```

自动入口只对当前目录所属形状运行对应的 `prompts`，不把所有形状轮流跑一遍。
同一图片的多个提示共享图像编码，重复 mask 会合并。每图最多保留一个主要物体：
优先选覆盖 `inference.target_point` 的候选，再比较靠近目标点的程度。
这适合现有“一张分类图片对应一个主要物体”的数据。
多个候选时记为 `multiple_candidates` 待复核；多物体、遮挡或目标位置不固定时使用交互入口。

结果保存到 `outputs_side/本次时间/`，包括：

- `dataset/`：原图片字节、YOLO-seg 多边形标签、类别表和 split 清单。
- `previews/`：轮廓、类别 ID、SAM 分数；中文类别见 `data.yaml`、`manifest.csv`。
- `manifest.csv`：记录原图片相对路径、导出名、具体类别和需要复核的原因。
- `summary.json`：图片数、实例数、零检出等统计。
- `sam3_cvat_project.zip`：在 CVAT Project 中使用 `Ultralytics YOLO Segmentation 1.0` 导入。

自动输出是候选标签。形状目录里的零检出是漏标，必须补标或剔除，不能直接当负样本训练。
断开的实例只保留最大连通轮廓，内孔会填平；预览中 `REVIEW` 的图片需要检查。
SAM 分数表示当前轮廓预测的分数，不是细分形状类别的可靠度。

自动批量结果可以直接用本工具的交互窗口复核，也可以导入 CVAT。

### 自动标一遍，再交互检查

1. 运行 `run_side_prelabel.cmd`，等待自动批次完成。
2. 运行 `run_side_interactive.cmd`。默认 `interactive.prelabels_dir: latest`，
   自动载入当前数据集最近完成批次的轮廓和具体类别。
3. 轮廓正确时，直接按 S 或 N 确认本张，不需要重新逐个点选。
4. 轮廓错误时，在实例列表选中并 Delete 删除，再用正负点、框或 P 手工多边形补标；
   类别错误时使用改类按钮。漏检图片会显示“自动零检出，待检查或补标”。
5. 逐张检查后按 E，导出的交互数据集只包含已由 S/N 确认的图片。

窗口里的“加载自动预标注批次……”按钮也可以手动选择 `outputs_side/本次时间/`，
或该批次内的 `dataset/` 子目录。修改 YAML 的 `interactive.prelabels_dir` 可指定某个批次；
设为 `null` 则新会话从零标注。恢复已有会话时优先沿用原来加载的批次，换批次使用按钮。

自动结果只作为草稿。已经人工保存的标签优先，加载新批次不会覆盖它们；
还没检查的图片不会因为导入或导出操作被当作已确认负样本。自动结果与人工结果分别保存。

## 4. 切换数据集和空背景

在 `sam3_side_config.yaml` 修改：

```yaml
dataset:
  source_dir: E:\smart_sorting_vision\data\datasets\side_cls\v2
  splits: [train, val, test]
```

如果换成没有 test 目录的 `school_test_new`，将 `splits` 改为 `[train, val]`。
交互标注换数据集时，同时修改 `interactive.output_dir`，避免混入另一批图片的记录。
`school_test_new` 当前只有 7 类，其余 5 类需要使用已有完整版本或另行采集。

不同形状或颜色下即使都叫 `image_000001.jpg`，也会按原相对路径生成稳定文件名后缀，
不会相互覆盖。原采集图片和目录保持原状。

现有侧面版本没有空台目录。需要增加负样本时，采集确认无目标的图片，放到
`train/no_object/`、`val/no_object/` 等目录。自动入口跳过这类图片的 SAM 推理、输出空标签；
交互入口显示空类别，确认空台后直接保存。**不增加一个 no_object 物体类别。**

原有数据大多是旧工位经过背景差分裁剪后的局部图。
它们可以复用来标轮廓，但正式模型若输入完整侧面识别台，需要补采新背景下的完整输入图，
覆盖目标位置变化、不同光照、贴二维码和空台场景。否则只换成分割模型也不能保证适应新背景。

## 5. 标签格式

每个物体一行，第一项是具体形状 ID，之后是归一化轮廓点：

```text
类别ID x1 y1 x2 y2 x3 y3 ...
```

例如第一个实例属于五棱柱，其 ID 由 `annotation.classes` 的顺序决定，
保存时会保留该 ID，不会统一写成顶部工具的 `0: object`。
输出图像不缩放、不重新编码，train/val/test 划分也保留。
