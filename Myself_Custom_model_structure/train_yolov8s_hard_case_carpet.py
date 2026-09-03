# hard_case训练命令(使用hard_case_file和hard_case_weight参数)
from ultralytics import YOLO

model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt")

model.train(
    task="segment",
    data="/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/datasets/hard_case_seg_carpet.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    hard_case_file="/data/database/AITotal_Segment_Hard_Case_Info/carpetDatabaseSegment/carpet_hard_cases.txt",
    hard_case_weight=5.0,
)
"""

# 普通训练命令(不使用hard_case_file和hard_case_weight参数)
from ultralytics import YOLO

model = YOLO(
    "/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt"
)

model.train(
    task="segment",
    data="/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/datasets/hard_case_seg_carpet.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
)
"""

"""

# 本地端训练命令
yolo segment train \
    model=/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt \
    data=/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/datasets/hard_case_seg_carpet.yaml \
    epochs=300 \
    imgsz=640 \
    batch=36 \
    workers=6 \
    amp=True \
    project=runs/my_carpet_seg_hard_case_exp \
    name=yolov8s_carpet_seg_hard_case_v1_ \
    augment=True \
    hard_case_file=/data/database/AITotal_Segment_Hard_Case_Info/carpetDatabaseSegment/carpet_hard_cases.txt \
    hard_case_weight=5.0
    device=0

# 服务器端训练命令
# 创建tmux项目 
tmux new -s yolov8_seg_training

yolo segment train \
    model=/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt \
    data=/workspace/data/TrainingScript/carpet_seg/seg_carpet.yaml \
    epochs=300 \
    imgsz=640 \
    batch=36 \
    workers=6 \
    amp=True \
    project=runs/my_carpet_seg_exp \
    name=yolov8s_carpet_seg_v1_ \
    augment=True \
    hard_case_file=/workspace/data/TrainingScript/carpet_seg/carpet_hard_cases.txt \
    hard_case_weight=5.0
    device=0

"""
