from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#


# 训练线材检测模型
model = YOLO(
    "/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_carpet.yaml"
)  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt")
model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
results = model.train(
    data="carpet_detect.yaml",
    epochs=300,
    imgsz=640,
    device=-1,
    workers=0,
    batch=32,
    project="runs/my_carpet_exp",
    name="yolov8_focus_v",
)


"""

# 创建tmux项目 
tmux new -s yolov8_seg_training

yolo segment train \
    model=/workspace/data/TrainingScript/carpet_seg/yolov8-seg_focus_carpet.yaml \
    data=/workspace/data/TrainingScript/carpet_seg/seg_carpet.yaml \
    epochs=300 \
    imgsz=640 \
    batch=36 \
    workers=6 \
    amp=True \
    project=runs/my_carpet_seg_exp \
    name=yolov8s_carpet_seg_v1_ \
    augment=True \
    multi_scale=True \
    device=0

"""
