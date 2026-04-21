"""
# ✅ Stage 1：只训练 head（稳定收敛）
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_10/weights/best.pt")

    model.train(
        task="segment",
        data="finetune_coco8-seg_carpet.yaml",

        epochs=50,
        imgsz=640,
        batch=16,

        device=0,
        workers=4,

        optimizer="SGD",
        lr0=0.001,
        cos_lr=True,

        freeze=10,              # ⭐ 冻 backbone

        mosaic=0.0,             # ⭐ 关闭强增强（关键）
        mixup=0.0,

        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        amp=False,

        project="runs/carpet_seg_finetune",
        name="stage1_head_only"
    )

"""


# ✅ Stage 2：全量 finetune（核心提升阶段）
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("runs/carpet_seg_finetune/stage1_head_only/weights/best.pt")

    model.train(
        task="segment",
        data="finetune_coco8-seg_carpet.yaml",

        epochs=80,
        imgsz=640,
        batch=8,                # ⭐ 稳一点

        device=0,
        workers=4,

        optimizer="SGD",
        lr0=3e-4,               # ⭐ 比三阶段稍高一点
        cos_lr=True,

        freeze=0,               # ⭐ 全解冻

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

        patience=40,

        project="runs/carpet_seg_finetune",
        name="stage2_full_finetune"
    )





