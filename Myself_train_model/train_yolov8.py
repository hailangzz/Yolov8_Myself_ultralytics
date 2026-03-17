from ultralytics import YOLO

# Load a model
model = YOLO(r"/yolov8s.pt")  # load a pretrained model (recommended for training)
results = model.train(data="wire_detect.yaml", epochs=100, imgsz=640, device=-1)

# model = YOLO(r"/home/chenkejing/PycharmProjects/ultralytics/runs/detect/train6/weights/last.pt")  # load a pretrained model (recommended for training)
# Train using the single most idle GPU
# results = model.train(data="wire_detect.yaml", epochs=100, imgsz=640, device=-1, resume=True)

