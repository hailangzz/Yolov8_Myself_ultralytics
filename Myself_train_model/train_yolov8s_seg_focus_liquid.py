from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_liquid.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#


# # 训练线材检测模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_carpet.yaml")  # load a pretrained model (recommended for training)
# # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# results = model.train(data="carpet_detect.yaml", epochs=300, imgsz=640, device=-1, workers=0, batch=32, project="runs/my_carpet_exp", name="yolov8_focus_v")
#

"""
对应的命令行代码：

yolo segment train \
    model=/workspace/data/TrainingScript/liquid_seg/yolov8-seg_focus_liquid.yaml \
    data=/workspace/data/TrainingScript/liquid_seg/seg_liquid.yaml \
    pretrained=/workspace/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_2/weights/last.pt \
    epochs=300 \
    imgsz=640 \
    batch=32 \
    workers=4 \
    amp=True \
    multi_scale=True \
    project=runs/my_liquid_seg_exp \
    name=yolov8s_liquid_seg_v1_ \
    augment=True \
    weight_decay=0.0005 \
    device=0
    
yolo segment train \
    model=/workspace/data/TrainingScript/liquid_seg/yolov8-seg_focus_liquid.yaml \
    data=/workspace/data/TrainingScript/liquid_seg/seg_liquid.yaml \
    epochs=300 \
    imgsz=640 \
    batch=32 \
    workers=4 \
    amp=True \
    project=runs/my_liquid_seg_exp \
    name=yolov8s_liquid_seg_v1_ \
    resume=True \
    augment=True \
    multi_scale=True \
    weight_decay=0.0005 \
    device=0
    
yolo segment train \
    model=/workspace/data/TrainingScript/liquid_seg/yolov8-seg_focus_liquid.yaml \
    data=/workspace/data/TrainingScript/liquid_seg/seg_liquid.yaml \
    epochs=300 \
    imgsz=640 \
    batch=32 \
    workers=4 \
    amp=True \
    project=runs/my_liquid_seg_exp \
    name=yolov8s_liquid_seg_v1_ \
    augment=True \
    multi_scale=True \
    weight_decay=0.0005 \
    device=0
    
    说明：
    amp=True 的作用：自动混合精度训练。带来的好处：显存减少 30~50%训练速度提升 20~60%可以用更大的 batch  
    workers=8 指的是 DataLoader 读取数据的并行进程数。CPU核心数推荐 8 推荐 workers=4; 核心数16 推荐workers=8
    augment=True :启动数据增强
    resume=True：自定加载，项目目录下的模型。与pretrained（手动指定预训练模型）一般不同时使用


# 线材小目标检测

 yolo segment train \
    model=/workspace/data/TrainingScript/liquid_seg/yolov8-seg_focus_liquid.yaml \
    data=/workspace/data/TrainingScript/liquid_seg/seg_liquid.yaml \
    epochs=300 \
    imgsz=1280 \
    batch=32 \
    workers=4 \
    amp=True \
    project=runs/my_liquid_seg_exp \
    name=yolov8s_liquid_seg_v1_rect_boxgain \
    augment=True \
    weight_decay=0.0005 \
    device=0 \
    box=2.0
       

    说明：
    rect=True
    启用长宽比训练，避免原图 1902×1080 被强制缩放到 640×640 导致横向压缩。
    保持线材的形状比例，提高小目标检测能力。
    box=2.0
    对 YOLOv8 来说，这个参数可以放大 box regression loss 的权重，对小目标更敏感。
    默认是 0.05~0.1 左右，你可以先试 2.0 或 1.5，看训练效果。
        
"""

from ultralytics import YOLO

if __name__ == "__main__":

    # 1️⃣ 加载分割模型结构（seg）
    model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8-seg_focus_liquid_0330.yaml")

    # 2️⃣ 加载预训练权重（非常重要）
    model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt")
    # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_2/weights/last.pt")

    # 3️⃣ 开始训练
    results = model.train(
        task="segment",                 # ⭐ 必须指定
        data="coco8-seg_liquid.yaml",      # 分割数据集 yaml
        epochs=300,
        imgsz=640,
        batch=30,                       # seg 比 detect 更吃显存
        device=0,                       # -1 = CPU，0 = GPU
        workers=0,

        optimizer="SGD",                # ⭐ 稳定，利于部署
        cos_lr=True,
        amp=False,                      # ⭐ ONNX/RKNN 强烈建议关  /amp 控制的是「训得快不快 vs 稳不稳」
        augment=True,
        dropout=0.1,
        project="runs/my_liquid_seg_exp",
        name="yolov8s_liquid_seg_v1_",
        # resume=False                    # 控制的是「训不训旧的状态」
        resume=True
    )

    # watch -n 1 nvidia-smi #监控GPU占用信息


