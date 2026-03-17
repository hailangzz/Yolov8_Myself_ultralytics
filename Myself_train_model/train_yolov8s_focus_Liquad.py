from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#

"""
yolo detect train \
    model=/workspace/data/TrainingScript/Liquad_detect/yolov8s_focus_Liquad.yaml \
    data=/workspace/data/TrainingScript/Liquad_detect/Liquad_detect.yaml \
    pretrained=yolov8s.pt \
    epochs=300 \
    imgsz=640 \
    batch=72 \
    workers=4 \
    amp=True \
    project=runs/my_Liquad_det_exp \
    name=yolov8s_Liquad_det_v1_ \
    augment=True \
    weight_decay=0.0005 \
    dropout=0.1 \
    device=0
    
    
"""

# 训练线材检测模型
model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_carpet.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt")
model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
results = model.train(data="carpet_detect.yaml", epochs=300, imgsz=640, device=-1, workers=0, batch=32, project="runs/my_carpet_exp", name="yolov8_focus_v")







