# PCB 缺陷检测

这是一个基于 **YOLOv8** 的 PCB 缺陷检测项目，旨在识别 PCB 电路板中的六种常见缺陷：`missing_hole`（漏孔）、`mouse_bite`（鼠咬）、`open_circuit`（开路）、`short`（短路）、`spur`（毛刺）和 `spurious_copper`（伪铜）。项目包含从数据预处理到模型训练的完整流程，适用于工业检测场景，展示了目标检测、数据增强和模型优化的实践。

## 代码调整：

1.当前代码是为 Kaggle 环境设计的，路径（如 /kaggle/input/pcb-defects/PCB_DATASET）和依赖安装（!pip install）需要调整为本地或通用环境。

2.将代码中的 !pip install 改为 requirements.txt，并调整路径为相对路径。

3.去掉 Kaggle 特定的输出日志（如训练过程中的 Epoch 输出），保留核心代码。

## 项目功能

- **数据集准备**：从原始图像和 XML 标注文件生成 YOLO 格式的训练和验证数据集。
- **标签转换**：将 XML 格式的标注转换为 YOLO 所需的 TXT 格式。
- **数据增强**：针对小目标缺陷（如 `spur` 和 `mouse_bite`），使用 **Albumentations** 进行增强，提升模型对小目标的检测能力。
- **模型训练**：基于 `yolov8m.pt` 预训练模型，训练 PCB 缺陷检测模型，优化参数以提高精度。
- **模型验证**：使用测试时增强（TTA）评估模型性能，输出 mAP@0.5 和 mAP@0.5:0.95。
- **模型导出**：将最佳模型导出为 ONNX 格式，便于跨平台部署。

## 依赖安装

运行以下命令安装所需依赖：
pip install -r requirements.txt

## 使用方法
1. 数据准备
将 PCB 数据集放置在项目根目录下的 PCB_DATASET 文件夹中，结构如下：

  PCB_DATASET/
  
  ├── images/  # 原始图像，按类别分文件夹
  
  └── Annotations/  # XML 标注文件，按类别分文件夹

运行脚本以生成训练和验证数据集：
python PCBDefectDetector.py

2. 训练模型

训练将在 PCB_DATASET/pcb_defects.yaml 配置下进行，结果保存在 runs/detect/train。

默认参数：50 个 epoch，图像大小 800，batch size 12。

3. 验证与导出

训练完成后，自动验证最佳模型并导出为 best.onnx。
