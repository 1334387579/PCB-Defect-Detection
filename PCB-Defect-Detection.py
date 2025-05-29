# %%capture --no-stderr
# 以下是一条Jupyter魔法命令，用于抑制此单元格的stderr输出。
# 对于隐藏冗长的pip安装输出很有用。
# 或者，先在单独的单元格中运行这些命令：
!pip install ultralytics -q
!pip install tqdm -q
!pip install opencv-python-headless pyyaml scikit-learn matplotlib numpy torch torchvision -q

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm # 在Kaggle notebooks中使用tqdm.notebook
import yaml # 用于创建dataset.yaml
import cv2 # OpenCV用于图像维度处理
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 0. 配置与设置
# -----------------------------------------------------------------------------
print("阶段 0: 配置与设置")

# --- 数据集配置 ---
BASE_INPUT_PATH = Path("/kaggle/input/pcb-dataset/PCB_DATASET")
IMAGE_DATA_PATH = BASE_INPUT_PATH / "images"
ANNOTATION_DATA_PATH = BASE_INPUT_PATH / "Annotations"

# --- YOLO 数据集输出配置 ---
# 为此优化运行使用新路径以保持输出分离
YOLO_DATASET_BASE_PATH = Path("/kaggle/working/pcb_yolo_dataset_v8_full_script_fix") 
YOLO_IMAGES_PATH = YOLO_DATASET_BASE_PATH / "images"
YOLO_LABELS_PATH = YOLO_DATASET_BASE_PATH / "labels"

# --- 类别定义 ---
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
print(f"已定义的类别: {CLASSES}")

# --- 训练配置 (针对更高分辨率和更多增强进行优化) ---
MODEL_NAME = 'yolov8m.pt'  # 中等模型。
EPOCHS = 100                # 训练周期数
# !!! 重要: BATCH_SIZE_PER_GPU 对于 yolov8m @ 1280 在单个T4上需要非常小 !!!
# 从1或2开始尝试。如果之前的运行（BATCH_SIZE_PER_GPU=16）没有OOM，
# 可能是Ultralytics内部动态调整了，但这里我们显式设置一个更合理的值。
BATCH_SIZE_PER_GPU = 16      # 为每个GPU设置的批处理大小 (对于yolov8m @ 1280 在T4上，2可能是一个起点)
IMG_SIZE = 1280             # 显著增加图像尺寸以更好地检测小目标
PATIENCE = 30               # 增加了早停的耐心值
ENABLE_AUGMENTATION = True  # 如果在train()中augment=True，则默认的Ultralytics增强将激活

PROJECT_NAME = 'pcb_defect_detection_v8_full_script_fix' 
RUN_NAME = f'{MODEL_NAME.split(".")[0]}_ep{EPOCHS}_b{BATCH_SIZE_PER_GPU}xgpu_img{IMG_SIZE}_aug_final'

# --- 详细的增强参数 (传递给 model.train()) ---
AUG_PARAMS = {
    'degrees': 10.0,       # 图像旋转 (+/- 度)
    'translate': 0.1,      # 图像平移 (+/- 比例)
    'scale': 0.5,          # 图像缩放 (+/- 增益)。对小物体过度缩小要小心。
    'shear': 2.0,          # 图像剪切 (+/- 度)
    'perspective': 0.0,    # 图像透视变换
    'flipud': 0.01,        # 图像垂直翻转 (概率)
    'fliplr': 0.5,         # 图像水平翻转 (概率)
    'mosaic': 1.0,         # 图像马赛克增强 (概率)
    'mixup': 0.1,          # 图像混合增强 (概率)，从0改为0.1尝试一下
    'copy_paste': 0.0,     # 片段复制粘贴增强 (概率)
    'hsv_h': 0.015,        # 图像HSV-Hue增强 (比例)
    'hsv_s': 0.7,          # 图像HSV-Saturation增强 (比例)
    'hsv_v': 0.4,          # 图像HSV-Value增强 (比例)
}


# --- 设备配置 ---
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用的GPU。")
    if num_gpus > 1:
        # 明确指定要使用的GPU ID列表
        device_setting_for_ultralytics = [i for i in range(num_gpus)] # 例如 [0, 1]
        print(f"将尝试使用GPU列表: {device_setting_for_ultralytics} 进行训练。")
        print(f"每个GPU的批处理大小为 {BATCH_SIZE_PER_GPU}，总有效批处理大小约为 {BATCH_SIZE_PER_GPU * num_gpus}。")
    elif num_gpus == 1:
        device_setting_for_ultralytics = 0 # 使用索引0的GPU
        print(f"将使用单个GPU (cuda:0) 进行训练。批处理大小为 {BATCH_SIZE_PER_GPU}。")
    else: 
        device_setting_for_ultralytics = 'cpu' # 理论上不应到这里
        print("警告: torch.cuda.is_available()为True但未检测到GPU？回退到CPU。")
    current_main_device = torch.device("cuda:0") # 主PyTorch设备用于模型加载等 (通常是第一个GPU)
else:
    device_setting_for_ultralytics = 'cpu'
    current_main_device = torch.device("cpu")
    print("未检测到GPU，将使用CPU进行训练。")

print(f"Ultralytics训练将使用的设备设置: {device_setting_for_ultralytics}")
print(f"当前 PyTorch 主设备 (用于评估/推理模型加载): {current_main_device}")
if current_main_device.type == 'cuda' and torch.cuda.is_available() and current_main_device.index is not None and current_main_device.index < torch.cuda.device_count():
     print(f"主GPU名称: {torch.cuda.get_device_name(current_main_device.index)}")
elif current_main_device.type == 'cuda' and torch.cuda.is_available(): # Fallback if index is None but cuda is available
     print(f"主GPU名称: {torch.cuda.get_device_name(0)}")


# -----------------------------------------------------------------------------
# 1. 数据准备辅助函数
# -----------------------------------------------------------------------------
print("\n阶段 1: 定义数据准备辅助函数")

def parse_xml_annotation(xml_file_path, img_path_for_fallback_dims):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_filename = root.find('filename').text
        size_node = root.find('size')
        image_width, image_height = None, None
        if size_node is not None:
            width_node, height_node = size_node.find('width'), size_node.find('height')
            if width_node is not None and width_node.text: image_width = int(width_node.text)
            if height_node is not None and height_node.text: image_height = int(height_node.text)
        
        if image_width is None or image_height is None: 
            try:
                img_cv = cv2.imread(str(img_path_for_fallback_dims))
                if img_cv is None: return None, None, None, [] 
                image_height, image_width, _ = img_cv.shape
            except Exception as e_cv:
                print(f"错误: 无法通过OpenCV获取图像 {xml_file_path.name} 的尺寸: {e_cv}。跳过此文件。")
                return None, None, None, []
        if image_width <= 0 or image_height <= 0: 
            print(f"错误: 图像 {xml_file_path.name} 的尺寸无效 ({image_width}x{image_height})。跳过此文件。")
            return None, None, None, []

        objects = []
        for obj_node in root.findall('object'): 
            class_name_from_xml = obj_node.find('name').text 
            if class_name_from_xml not in CLASSES: continue 
            bndbox_node = obj_node.find('bndbox') 
            try: 
                xmin, ymin, xmax, ymax = [int(float(bndbox_node.find(c).text)) for c in ['xmin', 'ymin', 'xmax', 'ymax']]
            except ValueError: continue 
            if xmin >= xmax or ymin >= ymax: continue 
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax) 
            if xmin >= xmax or ymin >= ymax: continue 
            objects.append({'name': class_name_from_xml, 'bndbox': [xmin, ymin, xmax, ymax]})
        return image_filename, image_width, image_height, objects
    except ET.ParseError as e_p: print(f"XML解析错误 {xml_file_path}: {e_p}。跳过。"); return None, None, None, []
    except Exception as e_g: print(f"处理XML文件 {xml_file_path} 时发生一般错误: {e_g}。跳过。"); return None, None, None, []

def voc_to_yolo(voc_bbox, img_width, img_height):
    if img_width <= 0 or img_height <= 0: return None 
    xmin, ymin, xmax, ymax = voc_bbox
    dw, dh = 1.0 / img_width, 1.0 / img_height 
    x_c, y_c = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0 
    w, h = xmax - xmin, ymax - ymin 
    if w <= 0 or h <= 0: return None 
    x_norm, y_norm, w_norm, h_norm = [np.clip(val, 0.0, 1.0) for val in [x_c * dw, y_c * dh, w * dw, h * dh]]
    if w_norm < 1e-6 or h_norm < 1e-6: return None 
    return x_norm, y_norm, w_norm, h_norm

def create_yolo_dataset_structure():
    if YOLO_DATASET_BASE_PATH.exists(): 
        print(f"清理已存在的YOLO数据集目录: {YOLO_DATASET_BASE_PATH}")
        shutil.rmtree(YOLO_DATASET_BASE_PATH)
    for split in ["train", "val", "test"]:
        (YOLO_DATASET_BASE_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_BASE_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("YOLO数据集目录结构已创建/重新创建。")

# -----------------------------------------------------------------------------
# 2. 数据处理与转换为YOLO格式
# -----------------------------------------------------------------------------
print("\n阶段 2: 数据处理与转换为YOLO格式")
create_yolo_dataset_structure() 
all_valid_img_paths, all_valid_lbl_data = [], [] 
stats = {'processed':0, 'no_xml':0, 'parse_err':0, 'no_valid_obj':0, 'xml_obj_found':0, 'yolo_obj_conv':0}
source_defect_folders = [d for d in IMAGE_DATA_PATH.iterdir() if d.is_dir()]
if not source_defect_folders: raise ValueError(f"在 {IMAGE_DATA_PATH} 中未找到缺陷子目录")
print(f"源缺陷文件夹: {[f.name for f in source_defect_folders]}")
for defect_path in tqdm(source_defect_folders, desc="处理源缺陷文件夹"):
    defect_name = defect_path.name 
    annot_path = ANNOTATION_DATA_PATH / defect_name 
    if not annot_path.exists(): print(f"警告: 未找到图像文件夹 {defect_path} 对应的标注文件夹 {annot_path}"); continue
    img_files = sorted(list(set(f for ext in ["*.jpg","*.png","*.jpeg"] for f in defect_path.glob(ext))))
    for img_p in tqdm(img_files, desc=f"处理 {defect_name} 中的图像", leave=False):
        stats['processed'] += 1 
        xml_p = annot_path / (img_p.stem + ".xml") 
        if not xml_p.exists(): stats['no_xml'] += 1; continue 
        fn, w, h, objs = parse_xml_annotation(xml_p, img_p) 
        if fn is None: stats['parse_err'] += 1; continue 
        stats['xml_obj_found'] += len(objs) 
        yolo_annots = [] 
        for obj in objs: 
            cls_id = CLASS_TO_ID.get(obj['name']) 
            if cls_id is None: continue 
            yolo_bb = voc_to_yolo(obj['bndbox'], w, h) 
            if yolo_bb is None: continue 
            yolo_annots.append(f"{cls_id} {yolo_bb[0]:.6f} {yolo_bb[1]:.6f} {yolo_bb[2]:.6f} {yolo_bb[3]:.6f}")
            stats['yolo_obj_conv'] +=1 
        if yolo_annots: 
            all_valid_img_paths.append(img_p) 
            all_valid_lbl_data.append(yolo_annots) 
        else: 
            stats['no_valid_obj'] += 1
print(f"\n--- 数据处理摘要 ---")
for k, v_stat in stats.items(): print(f"{k.replace('_',' ').capitalize()}: {v_stat}") 
print(f"保留的带有效标注的图像数量: {len(all_valid_img_paths)}")
if not all_valid_img_paths: raise ValueError("严重错误: 未处理任何带有效标注的图像。") 
paired_data = list(zip(all_valid_img_paths, all_valid_lbl_data))
random.seed(42); random.shuffle(paired_data) 
n_tot, n_tr, n_v = len(paired_data), int(len(paired_data)*0.7), int(len(paired_data)*0.15)
if n_tot > 0 and n_tr == n_tot : n_tr, n_v = (n_tot-1, 1) if n_tot > 1 else (n_tot, 0)
elif n_v == 0 and n_tot > n_tr: n_v = max(1, n_tot-n_tr)
if n_tr + n_v > n_tot: n_v = n_tot - n_tr 
train_d, val_d, test_d = paired_data[:n_tr], paired_data[n_tr:n_tr+n_v], paired_data[n_tr+n_v:]
print(f"数据集划分: {len(train_d)} 训练集, {len(val_d)} 验证集, {len(test_d)} 测试集.")
if not train_d: print("警告: 训练数据为空！训练很可能失败。")
if not val_d: print("警告: 验证数据为空！评估将被跳过或失败。")
def populate_yolo_split(data_split, name):
    if not data_split: print(f"'{name}' 数据集为空，跳过填充。"); return 0
    print(f"\n正在填充 '{name}' 数据集 ({len(data_split)} 张图像)...")
    img_r, lbl_r = YOLO_DATASET_BASE_PATH/"images"/name, YOLO_DATASET_BASE_PATH/"labels"/name
    c = 0
    for imp, yls in tqdm(data_split, desc=f"复制 {name} 文件"):
        try:
            shutil.copy(imp, img_r/imp.name) 
            with open(lbl_r/(imp.stem+".txt"), 'w') as f: [f.write(line+"\n") for line in yls] 
            c+=1
        except Exception as e: print(f"错误: 填充 {imp.name}到'{name}'时发生错误: {e}")
    print(f"完成填充 '{name}'。已复制 {c} 个图像-标签对。"); return c
populate_yolo_split(train_d, "train"); populate_yolo_split(val_d, "val"); populate_yolo_split(test_d, "test")
yaml_c = {'path':str(YOLO_DATASET_BASE_PATH.resolve()),'train':"images/train",'val':"images/val",'names':ID_TO_CLASS}
if test_d: yaml_c['test'] = "images/test" 
yaml_p = YOLO_DATASET_BASE_PATH / "dataset.yaml"
with open(yaml_p, 'w') as f: yaml.dump(yaml_c, f, sort_keys=False, indent=2) 
print(f"\nDataset YAML 文件: {yaml_p}"); [print(l_yaml,end='') for l_yaml in open(yaml_p,'r').readlines()] 
for s_split in ["train","val","test"]:
    ldir = YOLO_DATASET_BASE_PATH/"labels"/s_split
    if ldir.exists(): print(f"完整性检查 '{s_split}': {sum(1 for f_ in ldir.glob('*.txt') if f_.stat().st_size>0)} 非空 / {sum(1 for _ in ldir.glob('*.txt'))} 总计")

# -----------------------------------------------------------------------------
# 3. 模型训练
# -----------------------------------------------------------------------------
print("\n阶段 3: 模型训练")
train_imgs_dir = YOLO_DATASET_BASE_PATH / "images" / "train"
if not (train_imgs_dir.exists() and any(train_imgs_dir.iterdir())): 
    print("严重错误: 训练图像目录为空或不存在。无法开始训练。")
    results = None # 确保在无法训练时results为None
else:
    model = YOLO(MODEL_NAME) 
    print(f"开始训练: {MODEL_NAME}, 周期: {EPOCHS}, 每GPU批大小: {BATCH_SIZE_PER_GPU}, 图像尺寸: {IMG_SIZE}")
    print(f"启用增强: {ENABLE_AUGMENTATION}")
    if ENABLE_AUGMENTATION and any(AUG_PARAMS.values()): print(f"部分自定义增强参数将被使用。")
    elif ENABLE_AUGMENTATION: print("将使用Ultralytics的默认增强组合。")
    print(f"Dataset YAML: {yaml_p}, 项目: {PROJECT_NAME}, 运行名称: {RUN_NAME}")
    try:
        training_args = {
            'data': str(yaml_p), 
            'epochs': EPOCHS, 
            'batch': BATCH_SIZE_PER_GPU, 
            'imgsz': IMG_SIZE, 
            'patience': PATIENCE, 
            'device': device_setting_for_ultralytics, 
            'project': PROJECT_NAME, 
            'name': RUN_NAME, 
            'exist_ok': True, 
            'augment': ENABLE_AUGMENTATION 
        }
        if ENABLE_AUGMENTATION and any(AUG_PARAMS.values()): 
            print("正在合并自定义的增强参数。")
            training_args.update(AUG_PARAMS) 
        results = model.train(**training_args) 
        print("训练完成。")
    except Exception as e_train: 
        print(f"训练过程中发生错误: {e_train}"); 
        import traceback; 
        traceback.print_exc(); 
        results=None # 确保在训练出错时results为None
    
# -----------------------------------------------------------------------------
# 4. 模型评估
# -----------------------------------------------------------------------------
print("\n阶段 4: 模型评估")
best_model_p = None 

# 首先尝试从 results 对象 (如果训练成功且 results 非 None)
if results and hasattr(results, 'save_dir') and results.save_dir:
    potential_best_from_results = Path(results.save_dir) / "weights" / "best.pt"
    if potential_best_from_results.exists():
        best_model_p = potential_best_from_results
        print(f"从训练结果对象中找到最佳模型: {best_model_p}")

# 如果上述方法未找到，或者 results 为 None (例如训练未启动或中途失败)，则直接构造路径
if not best_model_p:
    # Ultralytics默认会将项目保存在当前工作目录下
    # PROJECT_NAME 和 RUN_NAME 定义了具体的子文件夹结构
    # 在Kaggle中，当前工作目录通常是 /kaggle/working/
    expected_save_dir = Path.cwd() / PROJECT_NAME / RUN_NAME # 使用 Path.cwd() 获取当前工作目录
    potential_best_direct = expected_save_dir / "weights" / "best.pt"
    if potential_best_direct.exists():
        best_model_p = potential_best_direct
        print(f"通过直接构造路径找到最佳模型: {best_model_p} (位于 {expected_save_dir})")
    else:
        print(f"尝试的直接路径未找到模型: {potential_best_direct}")
        # 最后的备用: 尝试从 model 对象中获取 (如果 model 对象存在且其 trainer 属性被填充)
        # 这通常在 results 为 None (例如训练因故未返回有效Results对象) 时可能有用，但 model 可能已部分训练
        if 'model' in locals() and hasattr(model, 'trainer') and hasattr(model.trainer, 'best') and model.trainer.best and Path(model.trainer.best).exists():
            best_model_p = Path(model.trainer.best)
            print(f"从model.trainer对象找到最佳模型: {best_model_p}")
        else:
            print(f"model.trainer.best 也不可用或路径无效。")

if best_model_p and best_model_p.exists(): 
    print(f"用于评估的最佳模型路径: {best_model_p}")
    eval_model = YOLO(best_model_p) 
    val_img_p = YOLO_DATASET_BASE_PATH / "images" / "val" 
    if val_img_p.exists() and any(val_img_p.iterdir()): 
        print("在验证集上评估最佳模型...")
        try:
            val_batch_for_eval = BATCH_SIZE_PER_GPU 
            eval_device_setting = device_setting_for_ultralytics
            if isinstance(device_setting_for_ultralytics, list) and device_setting_for_ultralytics:
                eval_device_setting = device_setting_for_ultralytics[0] 
            elif isinstance(device_setting_for_ultralytics, str) and ',' in device_setting_for_ultralytics:
                eval_device_setting = device_setting_for_ultralytics.split(',')[0]
            
            print(f"评估将在设备 '{eval_device_setting}' 上，批大小 {val_batch_for_eval} 进行。")
            metrics = eval_model.val(data=str(yaml_p), split='val', imgsz=IMG_SIZE, batch=val_batch_for_eval, device=eval_device_setting)
            if metrics and hasattr(metrics, 'box'): 
                print(f"\n验证指标: mAP50-95:{metrics.box.map:.4f}, mAP50:{metrics.box.map50:.4f}, mAP75:{metrics.box.map75:.4f}")
                if hasattr(metrics.box, 'maps') and metrics.box.maps is not None: 
                    print("\n各类别 mAP50-95:")
                    for i_map,cn_map in ID_TO_CLASS.items(): print(f"  {cn_map} (ID {i_map}): {metrics.box.maps[i_map]:.4f}" if i_map<len(metrics.box.maps) else f"  {cn_map} (ID {i_map}): N/A")
            else: print("验证结果中无指标或指标box属性。")
        except Exception as e_eval: print(f"模型验证出错: {e_eval}"); import traceback; traceback.print_exc()
    else: print("验证集图像目录为空。跳过评估。")
else: print("训练未成功或未找到有效的最佳模型路径。跳过评估。")

# -----------------------------------------------------------------------------
# 5. 推理/预测示例图像
# -----------------------------------------------------------------------------
print("\n阶段 5: 推理")
if best_model_p and best_model_p.exists(): 
    print(f"用于推理的最佳模型路径: {best_model_p}")
    pred_model = YOLO(best_model_p) 
    sample_img_p_inf = None
    val_imgs_for_inf_dir = YOLO_DATASET_BASE_PATH / "images" / "val" 
    if val_imgs_for_inf_dir.exists():
        try: sample_img_p_inf = next(val_imgs_for_inf_dir.glob("*.[jp][pn]g")) 
        except StopIteration: print("验证集中无样本图像用于推理。")
    
    if sample_img_p_inf and sample_img_p_inf.exists(): 
        print(f"\n在样本图像上推理: {sample_img_p_inf}")
        try:
            infer_device = '0' if torch.cuda.is_available() else 'cpu'
            if isinstance(device_setting_for_ultralytics, list) and device_setting_for_ultralytics:
                infer_device = str(device_setting_for_ultralytics[0])
            elif isinstance(device_setting_for_ultralytics, int): # Handles single GPU case
                 infer_device = str(device_setting_for_ultralytics)
            
            print(f"推理将在设备 '{infer_device}' 上进行。")
            preds_list_inf = pred_model.predict(source=str(sample_img_p_inf), save=True, conf=0.25, imgsz=IMG_SIZE, 
                                            project=PROJECT_NAME, name=RUN_NAME + "_inference_conf0.25", exist_ok=True, device=infer_device)
            if preds_list_inf: 
                res_obj_inf = preds_list_inf[0] 
                print(f"预测结果保存至: {res_obj_inf.save_dir}")
                img_plot_inf = res_obj_inf.plot() 
                plt.figure(figsize=(12,12)); plt.imshow(cv2.cvtColor(img_plot_inf, cv2.COLOR_BGR2RGB)) 
                plt.title(f"对 {sample_img_p_inf.name} 的预测 (置信度=0.25)"); plt.axis('off'); plt.show()
                print(f"\n在 {sample_img_p_inf.name} 中检测到的对象 (置信度 > 0.25):")
                if len(res_obj_inf.boxes) > 0: 
                    for box_inf in res_obj_inf.boxes: 
                        cid_inf, cname_inf = int(box_inf.cls), ID_TO_CLASS.get(int(box_inf.cls),"Unknown") 
                        cnf_inf, xy_inf = float(box_inf.conf), box_inf.xyxy[0].cpu().numpy().astype(int) 
                        print(f"  类别: {cname_inf} (ID {cid_inf}), 置信度: {cnf_inf:.3f}, 边界框: {xy_inf}")
                else: print("  无置信度大于0.25的检测结果。")
            else: print("模型预测返回空列表。")
        except Exception as e_pred: print(f"推理出错: {e_pred}"); import traceback; traceback.print_exc()
    else: print("未找到样本图像用于推理。")
else: print("未找到有效的最佳模型路径或训练失败。跳过推理。")

print("\n--- 脚本执行完毕 ---")