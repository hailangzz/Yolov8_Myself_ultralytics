from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#


# 训练线材检测模型
model = YOLO(
    "/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_hand.yaml"
)  # load a pretrained model (recommended for training)
model.load(
    "/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_hand_exp/yolov8_focus_v8/weights/best.pt"
)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
results = model.train(
    data="hand_detect.yaml",
    epochs=300,
    imgsz=416,
    device=-1,
    workers=0,
    batch=40,
    project="runs/my_hand_exp",
    name="yolov8_focus_v",
    resume=True,
)


# 5月21日 手势识别模型训练

"""
tmux new -s yolov8_seg_training
cd /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model

yolo detect train \
    model=/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_hand.yaml \
    data=/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/datasets/hand_detect.yaml \
    pretrained=/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_hand_exp/yolov8_focus_v7/weights/last.pt \
    epochs=300 \
    imgsz=416 \
    batch=40 \
    workers=4 \
    project=runs/my_hand_exp \
    name=yolov8_focus_v \
    augment=True \
    device=0
    
    

"""
