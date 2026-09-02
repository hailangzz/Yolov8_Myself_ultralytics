import json
import argparse
import pandas as pd
from ultralytics import YOLO


def eval_one_model(model_path, val_yaml):
    """评估单个模型，返回指标dict"""

    model = YOLO(model_path)
    metrics = model.val(data=val_yaml)

    # 自动兼容 detect / seg
    if hasattr(metrics, "seg") and metrics.seg is not None:
        mp = metrics.seg.mp
        mr = metrics.seg.mr
        map50 = metrics.seg.map50
        map5095 = metrics.seg.map
    else:
        mp = metrics.box.mp
        mr = metrics.box.mr
        map50 = metrics.box.map50
        map5095 = metrics.box.map

    precision = float(mp)
    recall = float(mr)

    return {
        "mAP50_95": float(map5095),
        "mAP50": float(map50),
        "recall": recall,
        "precision": precision,
        # ======================
        # 新增指标（关键）
        # ======================
        "miss_rate": 1 - recall,  # 漏检率
        "false_alarm_rate": 1 - precision  # 误检率
    }


def metrics_model_performance(model_paths, val_yaml, save_name):
    """
    model_paths: list[str]
    """

    all_results = {}

    for model_path in model_paths:
        print(f"\n🚀 Evaluating: {model_path}")

        metrics_dict = eval_one_model(model_path, val_yaml)

        # 用模型文件名作为列名
        model_name = model_path.split("/")[-3].replace(".pt", "")

        all_results[model_name] = metrics_dict

        print(f"✔ {model_name}: {metrics_dict}")

    # =========================
    # 转为 CSV
    # =========================
    df = pd.DataFrame(all_results)

    csv_path = save_name + ".csv"
    df.to_csv(csv_path, index=True, encoding="utf-8-sig")

    print(f"\n📊 所有模型评估完成，结果已保存到：{csv_path}")
    print(df)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Multi-model Evaluation")

    # 关键修改：支持多个模型路径
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of model paths"
    )

    parser.add_argument(
        "--val_yaml",
        type=str,
        required=True,
        help="Validation yaml file"
    )

    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Output CSV file name"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metrics_model_performance(
        args.model_paths,
        args.val_yaml,
        args.save_name
    )

    # 模型性能评估，命令行脚本
    """
    # 地毯检测模型效果测试

    python \
    Myself_Model_Val_Metrics/predict_seg_images_target_classify.py \
    --model_paths \
    Myself_train_finetune_model/runs/carpet_seg_finetune/stage1_head_only/weights/best.pt\
    Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune/weights/best.pt \
    Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune2/weights/best.pt \
    --val_yaml \
    Myself_Model_Val_Metrics/carpet_val.yaml \
    --save_name \
    eval_compare



    """
