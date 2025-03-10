# %% [1] 安装依赖
!pip install ultralytics split-folders opencv-python-headless albumentations --quiet
# %% [2] 初始化设置
import os
import shutil
import xml.etree.ElementTree as ET
import torch
import cv2
import numpy as np
from tqdm import tqdm
import splitfolders
from ultralytics import YOLO
import albumentations as A

# 配置参数
class Config:
    def __init__(self):
        self.class_names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
        self.input_path = "/kaggle/input/pcb-defects/PCB_DATASET"
        self.output_path = "/kaggle/working/PCB_DATASET"
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.seed = 42
        self.img_size = 800  # 降低图像尺寸以减少内存使用
        self.epochs = 50
        self.batch_size = 12  # 降低 batch size 以减少内存使用
        self.model_type = "yolov8m.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augment_targets = ["spur", "mouse_bite"]

cfg = Config()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# %% [3] 数据集准备
def prepare_dataset():
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        full_path = os.path.join(cfg.output_path, d)
        if os.path.exists(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
        os.makedirs(full_path, exist_ok=True)
    
    splitfolders.ratio(
        os.path.join(cfg.input_path, 'images'),
        output=os.path.join(cfg.output_path, '_temp'),
        seed=cfg.seed,
        ratio=(cfg.train_ratio, cfg.val_ratio),
        group_prefix=None,
        move=False
    )
    
    for split in ['train', 'val']:
        source_dir = os.path.join(cfg.output_path, '_temp', split)
        target_dir = os.path.join(cfg.output_path, 'images', split)
        for class_name in os.listdir(source_dir):
            class_source_dir = os.path.join(source_dir, class_name)
            class_target_dir = os.path.join(target_dir, class_name)
            if os.path.isdir(class_source_dir):
                shutil.copytree(class_source_dir, class_target_dir, dirs_exist_ok=True)
    
    shutil.rmtree(os.path.join(cfg.output_path, '_temp'), ignore_errors=True)

prepare_dataset()

# %% [4] 标签转换与增强
class DatasetBuilder:
    def __init__(self):
        self.class_ids = {name: idx for idx, name in enumerate(cfg.class_names)}
        self.aug_transform = A.Compose([
            A.GridDropout(ratio=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomScale(scale_limit=0.3, p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2))
      
    def convert_labels(self, img_dir, xml_dir, label_dir):
        for class_name in os.listdir(img_dir):
            img_class_dir = os.path.join(img_dir, class_name)
            if not os.path.isdir(img_class_dir):
                continue
            
            xml_class_dir = os.path.join(xml_dir, class_name)
            label_class_dir = os.path.join(label_dir, class_name)
            os.makedirs(label_class_dir, exist_ok=True)
            print(f"Processing class: {class_name}, img_dir: {img_class_dir}, xml_dir: {xml_class_dir}, label_dir: {label_class_dir}")
            
            for img_file in tqdm(os.listdir(img_class_dir), desc=f"Processing {class_name}"):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
              
                base_name = os.path.splitext(img_file)[0]
                xml_path = os.path.join(xml_class_dir, f"{base_name}.xml")
                txt_path = os.path.join(label_class_dir, f"{base_name}.txt")
              
                if not os.path.exists(xml_path):
                    print(f"Warning: XML file not found for {img_file}, skipping.")
                    continue
              
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find('size')
                    if size is None:
                        print(f"Warning: No size element in {xml_path}, skipping.")
                        continue
                    w = int(size.find('width').text)
                    h = int(size.find('height').text)
                    if w <= 0 or h <= 0:
                        print(f"Warning: Invalid size in {xml_path} (w={w}, h={h}), skipping.")
                        continue
                  
                    with open(txt_path, 'w') as f:
                        for obj in root.findall('object'):
                            cls_name = obj.find('name').text.strip().lower()
                            if cls_name not in self.class_ids:
                                print(f"Warning: Unknown class {cls_name} in {xml_path}, skipping object.")
                                continue
                          
                            bbox = obj.find('bndbox')
                            if bbox is None:
                                print(f"Warning: No bndbox in {xml_path}, skipping object.")
                                continue
                            xmin = max(0, float(bbox.find('xmin').text))
                            ymin = max(0, float(bbox.find('ymin').text))
                            xmax = min(w, float(bbox.find('xmax').text))
                            ymax = min(h, float(bbox.find('ymax').text))
                            if xmin >= xmax or ymin >= ymax:
                                print(f"Warning: Invalid bbox coordinates in {xml_path}, skipping object.")
                                continue
                          
                            x_center = ((xmin + xmax) / 2) / w
                            y_center = ((ymin + ymax) / 2) / h
                            width = (xmax - xmin) / w
                            height = (ymax - ymin) / h
                          
                            f.write(f"{self.class_ids[cls_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                          
                except ET.ParseError as e:
                    print(f"Error parsing XML {xml_path}: {str(e)}")
                except Exception as e:
                    print(f"Error processing {xml_path}: {str(e)}")
  
    def augment_small_objects(self, img_dir, label_dir):
        for class_name in cfg.augment_targets:
            class_img_dir = os.path.join(img_dir, class_name)
            class_label_dir = os.path.join(label_dir, class_name)
            
            if not os.path.exists(class_img_dir) or not os.path.exists(class_label_dir):
                print(f"Warning: Directory {class_img_dir} or {class_label_dir} not found, skipping.")
                continue
            
            for img_file in tqdm(os.listdir(class_img_dir), desc=f"Augmenting {class_name}"):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(class_img_dir, img_file)
                label_path = os.path.join(class_label_dir, os.path.splitext(img_file)[0] + ".txt")

                if not os.path.exists(label_path):
                    print(f"Warning: Label file not found for {img_file}, skipping.")
                    continue

                with open(label_path) as f:
                    labels = [list(map(float, line.strip().split())) for line in f if line.strip()]

                if not labels:
                    print(f"Warning: No labels found in {label_path}, skipping.")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Failed to load image {img_path}, skipping.")
                    continue
                h, w = img.shape[:2]

                augmented = self.aug_transform(image=img, bboxes=labels)
                new_img = augmented['image']
                new_labels = augmented['bboxes']

                new_name = f"aug_{img_file}"
                cv2.imwrite(os.path.join(class_img_dir, new_name), new_img)
                with open(os.path.join(class_label_dir, new_name.replace('.jpg', '.txt')), 'w') as f:
                    for label in new_labels:
                        f.write(" ".join(map(str, [int(label[0])] + label[1:])) + "\n")

# 检查数据集完整性
def check_dataset(img_dir, label_dir):
    for class_name in os.listdir(img_dir):
        class_img_dir = os.path.join(img_dir, class_name)
        class_label_dir = os.path.join(label_dir, class_name)
        
        if not os.path.exists(class_img_dir) or not os.path.exists(class_label_dir):
            continue
        
        for img_file in os.listdir(class_img_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            label_file = os.path.join(class_label_dir, os.path.splitext(img_file)[0] + ".txt")
            if not os.path.exists(label_file):
                print(f"Error: Missing label file for {img_file}")
                continue
            with open(label_file) as f:
                lines = f.readlines()
                if not lines:
                    print(f"Error: Empty label file {label_file}")

# 执行转换和增强
builder = DatasetBuilder()
for split in ['train', 'val']:
    builder.convert_labels(
        os.path.join(cfg.output_path, f'images/{split}'),
        os.path.join(cfg.input_path, 'Annotations'),
        os.path.join(cfg.output_path, f'labels/{split}')
    )
builder.augment_small_objects(
    os.path.join(cfg.output_path, 'images/train'),
    os.path.join(cfg.output_path, 'labels/train')
)

# 检查数据集
check_dataset(
    os.path.join(cfg.output_path, 'images/train'),
    os.path.join(cfg.output_path, 'labels/train')
)
check_dataset(
    os.path.join(cfg.output_path, 'images/val'),
    os.path.join(cfg.output_path, 'labels/val')
)

# %% [5] 创建数据集配置文件
dataset_yaml = f"""
path: {cfg.output_path}
train: images/train
val: images/val
nc: 6
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""

with open(os.path.join(cfg.output_path, "pcb_defects.yaml"), "w") as f:
    f.write(dataset_yaml)

# %% [6] 模型训练
model = YOLO(cfg.model_type)

# 预生成数据集缓存
print("Pre-generating dataset cache...")
model.val(
    data=os.path.join(cfg.output_path, "pcb_defects.yaml"),
    imgsz=cfg.img_size,
    batch=cfg.batch_size,
    workers=0,
    cache=True
)
print("Dataset cache generated successfully.")

# 启动训练
train_params = {
    'data': os.path.join(cfg.output_path, "pcb_defects.yaml"),
    'epochs': cfg.epochs,
    'imgsz': cfg.img_size,
    'batch': cfg.batch_size,
    'device': cfg.device,
    'optimizer': 'AdamW',
    'lr0': 0.002,
    'lrf': 0.01,
    'cos_lr': True,
    'augment': True,
    'mosaic': 0.5,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'degrees': 30,
    'translate': 0.3,
    'scale': 0.7,
    'shear': 13.0,
    'perspective': 0.001,
    'fliplr': 0.7,
    'cls': 0.1,
    'box': 0.2,
    'dfl': 0.2,
    'close_mosaic': 20,
    'patience': 50,
    'amp': False,
    'weight_decay': 0.0005,
    'nbs': 128,
    'workers': 0
}

try:
    results = model.train(**train_params)
except OSError as e:
    print(f"Caught OSError: {e}. Attempting to reinitialize and retry...")
    model = YOLO(cfg.model_type)
    results = model.train(**train_params)
except Exception as e:
    print(f"Unexpected error: {e}")
    raise

# %% [7] 模型验证与优化
best_model = YOLO(os.path.join(model.trainer.save_dir, 'weights', 'best.pt'))

# TTA验证
metrics = best_model.val(
    batch=cfg.batch_size * 2,
    conf=0.25,
    iou=0.45,
    imgsz=cfg.img_size,
    plots=True,
    augment=True
)

print(f"验证结果 mAP@0.5: {metrics.box.map50:.4f}")
print(f"验证结果 mAP@0.5:0.95: {metrics.box.map:.4f}")

# 导出ONNX
best_model.export(
    format='onnx',
    dynamic=True,
    simplify=True,
    opset=12,
    imgsz=cfg.img_size,
    device='cpu'
)

# %% [8] 清理与归档
shutil.rmtree(os.path.join(cfg.output_path, '_temp'), ignore_errors=True)
print("训练完成！模型路径:", model.trainer.save_dir)
