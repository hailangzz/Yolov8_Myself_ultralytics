import json
from ultralytics import YOLO
import argparse


def metrics_model_performance(model_path, val_yaml, save_name):
    # 加载你的模型
    # model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_wire_exp/yolov8_focus_sa_v28/weights/best.pt")
    model = YOLO(model_path)

    # 评估模型
    # metrics = model.val(data="wire_test.yaml")
    metrics = model.val(data=val_yaml)
    print(metrics)
    # 转成 Python dict（metrics 是 ultralytics 对象，需要转换）
    results_dict = {
        "precision": float(metrics.box.mp),         # P
        "recall": float(metrics.box.mr),            # R
        "mAP50": float(metrics.box.map50),          # mAP@50
        "mAP50_95": float(metrics.box.map),         # mAP@50-95
    }
    print(results_dict)
    # 保存到本地 JSON 文件
    # save_path = "eval_wire_result_model8_foucs_sa.json"
    save_path = save_name+".json"
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"测试结果已保存到：{save_path}")
    print(results_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained YOLO model")
    parser.add_argument("--val_yaml", type=str, required=True, help="Directory the images to val")
    parser.add_argument("--save_name", type=str, required=True, help="Directory to save the metrics results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_model_performance(args.model_path, args.val_yaml, args.save_name)

    #python metrics_test.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_wire_exp/yolov8_focus_sa_v28/weights/best.pt --val_yaml wire_test.yaml --save_name eval_wire_result_model8_foucs_sa
