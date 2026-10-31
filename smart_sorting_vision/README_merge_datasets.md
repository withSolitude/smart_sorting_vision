# 合并补充标注的数据集

此工具合并**已经标注并导出的 YOLO 检测或实例分割数据集**，不加载模型、不需要GPU。
只有照片的采集目录需要先完成 SAM3 标注和人工复查，再导出。
分类任务的 `train/类别/照片` 目录不使用本工具。

## 使用

打开项目根目录的 `merge_datasets.yaml`。数据集放在同一个根目录时，平时只改两个名字：

```yaml
dataset_root: data/datasets/top_seg
dataset_a: v2_exports
dataset_b: v3_exports
```

名称相对 `dataset_root`；`dataset_root` 相对脚本所在的视觉工程目录。
也可以只把某个名字改成绝对路径，例如：

```yaml
dataset_b: E:/gongxun/27/sam3/outputs_interactive_v3
```

可选择含 `data.yaml` 的数据集目录、SAM3 输出目录（自动寻找 `dataset/data.yaml`），或直接选择 `data.yaml`。
支持 YAML 的 train/val/test 指向图片目录、TXT清单或路径列表。

在项目目录运行：

```bash
python merge_datasets.py
```

需要的依赖只有工程已有的 PyYAML；若提示 `No module named yaml`：

```bash
python -m pip install pyyaml
```

每次输出一个新目录：

```text
data/datasets/top_seg/merged/两个名字_时间戳/
  data.yaml
  classes.txt
  train.txt / val.txt / test.txt
  images/train/...
  labels/train/...
  images/val/... / labels/val/...
  images/test/... / labels/test/...
  merge_manifest.csv
  summary.json
```

终端会打印完整输出路径，训练脚本选择该目录下的 `data.yaml`。
图片保留原始字节，不改变分辨率；图片与标签成对重命名，防止同名覆盖。
输出使用相对清单，可将整个输出目录一起复制到Jetson。原数据不修改。

## 合并规则

- 保留原 train/val/test，不随机重新划分。未提供的 split 输出为空清单。
- 按类别名称对齐编号，A 的类别顺序不变，B 中同名类别重映射；新增类别追加。
- 类别名称代表同一含义、顶面/整物体等标注规则一致，需要在输入数据中先确认。
- 相同图片字节、相同 split、相同标注：只保留一份，记录到 manifest。
- 相同图片跨 split：报出冲突，不自动改变归属；需先从一个源集合移除重复图及对应标签。
- 相同图片存在不同标注：报错，先人工确定正确的一份。
- 空 TXT 表示已确认的无目标图片，会保留；缺 TXT 表示未标注，报错而不当作背景。
- 支持检测框或分割多边形，但一个输出不能混合这两种非空标签。

去重只识别完全相同的图片文件，不能识别近似连拍；近似场景应在采集时保持同一个 split。
`merge_manifest.csv` 记录每张输入图的原路径、新路径及是否去重，`summary.json` 记录数量和类别。
SAM3 的 `annotations.json` 属于人工交互会话，本工具不会合并它；旧会话与新会话继续分别保留。
