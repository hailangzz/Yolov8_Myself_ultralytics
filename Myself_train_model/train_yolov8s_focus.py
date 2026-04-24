from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#


# 训练线材检测模型
model = YOLO(
    "/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml"
)  # load a pretrained model (recommended for training)
model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
results = model.train(
    data="wire_detect.yaml",
    epochs=100,
    imgsz=640,
    device=0,
    workers=0,
    resume=True,
    project="runs/my_wire_exp",
    name="yolov8_focus_v1",
)
