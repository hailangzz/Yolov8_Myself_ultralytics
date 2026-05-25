## Stage 1：只训练 Head（稳定收敛）
## 👉 目标：
## 保留 backbone 特征
## 只让检测头适配新数据
## 防止一开始就破坏已有能力

"""
from ultralytics import YOLO.

if __name__ == "__main__":

    model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_hand_exp/yolov8_focus_v7/weights/best.pt")

    model.train(
        task="detect",
        data="finetune_hand_detect.yaml",

        epochs=50,
        imgsz=416,
        batch=40,

        device=0,
        workers=4,

        optimizer="SGD",
        lr0=0.001,
        cos_lr=True,

        freeze=10,              # ⭐ 冻 backbone（关键）

        mosaic=0.0,             # ⭐ 关闭强增强（关键）
        mixup=0.0,
        copy_paste=0.0,

        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        amp=False,

        project="runs/my_hand_finetune",
        name="stage1_head_only"
    )
"""


# """
## Stage 2：全量 Finetune（核心提升阶段）
## 👉 目标：
## 解冻 backbone
## 提升整体性能（精度、泛化）

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/my_hand_finetune/stage1_head_only/weights/best.pt")

    model.train(
        task="detect",
        data="finetune_hand_detect.yaml",
        epochs=150,
        imgsz=416,
        batch=32,  # ⭐ 稍微降一点更稳
        device=0,
        workers=4,
        optimizer="SGD",
        lr0=3e-4,  # ⭐ finetune 常用学习率
        cos_lr=True,
        freeze=0,  # ⭐ 全解冻
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        weight_decay=0.0005,
        amp=False,
        patience=50,
        project="runs/my_hand_finetune",
        name="stage2_full_finetune",
    )

# """
