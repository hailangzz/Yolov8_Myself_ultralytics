from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#

"""
from ultralytics import YOLO

# 加载自定义模型结构
model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_sa_hand_v3.yaml")

# 加载已有权重
model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v3_/weights/last.pt")

# 设置超参数，增加正则化
hyp = {
    'weight_decay': 0.0005,  # L2正则化
    'dropout': 0.1           # Dropout 比例
}

# 开始训练，增加早停
results = model.train(
    data="hand_detect.yaml",
    epochs=300,
    imgsz=416,
    device=-1,
    workers=0,
    batch=60,
    project="runs/my_hand_exp",
    name="yolov8_focus_sa_v3_",
    resume=True,
    hyp=hyp,                # 超参数
    augment=True,           # 数据增强
    early_stopping=50       # 若验证集连续50个epoch没有提升则提前停止
)

对应的命令行代码：

yolo detect train \
    model=/workspace/data/TrainingScript/yolov8s_focus_sa_hand_v3.yaml \
    data=/workspace/data/TrainingScript/hand_detect.yaml \
    pretrained=/workspace/data/TrainingScript/last.pt \
    epochs=300 \
    imgsz=416 \
    batch=120 \
    workers=0 \
    project=runs/my_hand_exp \
    name=yolov8_focus_sa_v3_ \
    resume=True \
    augment=True \
    weight_decay=0.0005 \
    dropout=0.1 \
    device=0

"""

"""
from ultralytics import YOLO

# -----------------------------
# 1️⃣ 配置模型与权重
# -----------------------------
model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_sa_hand_v3.yaml")

# 加载已有训练权重
model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v3_/weights/last.pt")

# -----------------------------
# 2️⃣ 定义训练参数
# -----------------------------
train_params = {
    "data": "hand_detect.yaml",   # 数据集配置
    "epochs": 300,                # 最大训练轮次
    "imgsz": 416,                 # 输入图像尺寸
    "device": -1,                 # 使用GPU，-1表示自动选择
    "workers": 4,                 # 数据加载线程数
    "batch": 60,                  # batch size
    "project": "runs/my_hand_exp",
    "name": "yolov8_focus_sa_v3_reg",
    "resume": True,               # 继续训练
    # ---------- 学习率与优化器 ----------
    "lr0": 0.001,                 # 初始学习率，较原训练降低
    "lrf": 0.01,                  # 最终学习率比例
    # ---------- 正则化 ----------
    "weight_decay": 0.0005,       # L2正则化
    "dropout": 0.1,               # 在模型中适当启用dropout
    # ---------- 数据增强 ----------
    "augment": True,              # 启用默认增强
    # ---------- 早停 ----------
    "early_stopping": 30,         # 若30轮验证集loss不下降则停止
}

# -----------------------------
# 3️⃣ 启动训练
# -----------------------------
results = model.train(**train_params)

# -----------------------------
# 4️⃣ 保存最终模型
# -----------------------------
results.model.save("runs/my_hand_exp/yolov8_focus_sa_v3_reg_final.pt")

"""

# # 训练线材检测模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_sa_hand_v2.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v/weights/last.pt")
# results = model.train(data="hand_detect.yaml", epochs=300, imgsz=416, device=-1, workers=0, batch=40, project="runs/my_hand_exp", name="yolov8_focus_sa_v", resume=True)

# 训练线材检测模型
model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_sa_hand_v3.yaml")  # load a pretrained model (recommended for training)
model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v3_/weights/last.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
results = model.train(data="hand_detect.yaml", epochs=300, imgsz=416, device=-1, workers=0, batch=60, project="runs/my_hand_exp", name="yolov8_focus_sa_v3_")






