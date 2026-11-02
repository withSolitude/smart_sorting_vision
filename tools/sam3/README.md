# SAM 3 分拣物料预标注工具

使用官方 SAM 3 本地权重，对已有 train/val/test 图片递归预标注，
导出 YOLO-seg 标签、预览图和 CVAT 导入包。顶部使用单类别 `object`，侧面使用具体形状类别。

完整操作说明见 [README_SAM3_使用说明.md](README_SAM3_使用说明.md)。
几何体与顶面标注规则见 [几何体与顶面标注说明.md](几何体与顶面标注说明.md)。
正负点交互入口见 [README_交互式标注.md](README_交互式标注.md)，本机双击 `run_interactive.cmd`。
侧面12类入口见 [README_侧面分割预标注.md](README_侧面分割预标注.md)：
双击 `run_side_interactive.cmd` 点选标注，或 `run_side_prelabel.cmd` 自动生成候选。
侧面参数集中在 `sam3_side_config.yaml`，类别从现有形状目录读取。
支持先批量自动标注、再用交互窗口复核。`interactive.prelabels_dir: latest`
会加载当前数据集最近完成的自动批次；也可在窗口中选择指定批次。

## 从 GitHub 下载后的准备

仓库只保存接口、配置和启动脚本，不包含本机环境、图片、输出、官方源码 ZIP 或模型权重。

1. 从 [官方仓库](https://github.com/facebookresearch/sam3) 取得源码，放到本目录的
   `sam3-main/`，确保存在 `sam3-main/sam3/model_builder.py` 和官方分词词表。
2. 按官方模型许可取得权重，放到本目录的 `sam3.pt`。本地已有文件可以直接复制。
3. 使用已安装匹配 CUDA 版 PyTorch/torchvision 的 Python，执行 `install_sam3.cmd`。
   本工具不会安装或替换 PyTorch。当前接口使用 NVIDIA CUDA 和 BF16。
4. 修改 `sam3_config.yaml` 中的 `dataset.source_dir`；没有的 split 从 `splits` 中移除。
   模型、输出路径相对于 YAML 所在目录解析。
5. 在 PowerShell 中运行：

```powershell
.\.venv_sam3\Scripts\python.exe prelabel_sam3.py --dry-run
.\run_prelabel.cmd --limit 5
```

本机原工具目录为 `E:\gongxun\27\sam3`，环境和权重已在这里准备。
此路径只说明本机位置，脚本不依赖这个绝对位置。

## 当前版本的含义

原顶部入口的 `inference.prompts` 是 SAM 3 的文字提示，每个提示独立推理，再按 mask IoU 去重。
提示名称不会变成训练类别：所有实例仍导出为类别 `0: object`。
数据集输入保留本机的 `top_seg/v2`，批量入口读取你当前填写的文字提示。
交互入口不依赖文字提示，用鼠标指定目标区域；标完整物体还是顶面由人工确认。

提示词只是候选生成条件，不保证排除托盘或只分割顶面。
批量运行前先检查少量预览；最终标签需要人工修正。
