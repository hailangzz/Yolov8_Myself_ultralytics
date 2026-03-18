import os

from ultralytics import YOLO

# ---------------------------
# 配置参数
# ---------------------------
# 已训练好的模型路径
PRETRAINED_MODEL = "./runs/detect/exp_best.pt"  # 替换为你的模型路径

# 新数据集路径（包含原数据 + 误检样本）
DATASET_YAML = "./dataset/dataset_with_misdet.yaml"  # YOLO 数据集 YAML 文件

# 输出路径
OUTPUT_DIR = "./runs/fine_tune"

# 超参数
EPOCHS = 50  # 可以根据新数据量调整
BATCH_SIZE = 16  # 根据 GPU 调整
LEARNING_RATE = 0.001  # 较原训练小的学习率
IMG_SIZE = 640  # 输入尺寸

# ---------------------------
# 创建输出目录
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 加载预训练模型
# ---------------------------
model = YOLO(PRETRAINED_MODEL)

# ---------------------------
# 开始微调训练
# ---------------------------
model.train(
    data=DATASET_YAML,  # 数据集 YAML
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    lr0=LEARNING_RATE,  # 初始学习率
    project=OUTPUT_DIR,
    name="fine_tune",
    exist_ok=True,  # 如果目录存在则覆盖
    pretrained=True,  # 使用预训练权重
    freeze=[0],  # 可选：冻结 backbone（第0个模块），只训练 head
    workers=8,
)

# ---------------------------
# 保存最终权重
# ---------------------------
final_weights = os.path.join(OUTPUT_DIR, "fine_tune", "weights", "best.pt")
print(f"Fine-tune finished. Best model saved at: {final_weights}")
