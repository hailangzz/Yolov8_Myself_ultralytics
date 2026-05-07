from ultralytics import YOLO

# 训练地毯识别模型
model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8-seg_focus_cross_database.yaml")  # load a pretrained model (recommended for training)
model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
results = model.train(data="cross_database_multis-seg.yaml", epochs=3, imgsz=640, device=-1, workers=0, batch=32, project="runs/my_cross_database_exp", name="yolov8_focus_v")
