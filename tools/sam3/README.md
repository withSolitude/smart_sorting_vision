# SAM 3 分拣物料预标注工具

使用官方 SAM 3 本地权重，对已有 train/val/test 图片递归预标注，
导出单类别 `object` 的 YOLO-seg 标签、预览图和 CVAT 导入包。

完整操作说明见 [README_SAM3_使用说明.md](README_SAM3_使用说明.md)。
几何体与顶面标注规则见 [几何体与顶面标注说明.md](几何体与顶面标注说明.md)。

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

`inference.prompts` 是 SAM 3 的文字提示，每个提示独立推理，再按 mask IoU 去重。
提示名称不会变成训练类别：所有实例仍导出为类别 `0: object`。
当前默认配置保留颜色提示，数据集输入保留本机的 `top_seg/v2`。
本次上传不自动改为顶面任务。

提示词只是候选生成条件，不保证排除托盘或只分割顶面。
批量运行前先检查少量预览；最终标签需要人工修正。
