# -*- coding: utf-8 -*-

"""
YOLOv8-seg Model Benchmark

功能：
1. 在同一个 YOLO validation dataset 上测试多个 YOLOv8-seg 模型
2. 统计：
   - Mask Precision
   - Mask Recall
   - Mask F1
   - Mask mAP50
   - Mask mAP50-95
   - Box Precision
   - Box Recall
   - Box mAP50
   - Box mAP50-95
3. 统计各类别 Mask Precision / Recall / mAP
4. 统计推理速度
5. 输出 CSV / JSON
6. 终端打印模型横向对比结果

使用：

python benchmark.py \
    --data dataset/data.yaml \
    --models models/yolov8n_seg_v1.pt \
             models/yolov8n_seg_v2.pt \
             models/yolov8n_seg_v3.pt \
    --imgsz 640 \
    --batch 16 \
    --device 0

也可以直接指定模型目录：

python benchmark.py \
    --data dataset/data.yaml \
    --model-dir models \
    --imgsz 640 \
    --batch 16 \
    --device 0
"""

import argparse
import csv
import json
import time
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# 工具函数
# ============================================================


def safe_float(value):
    """
    安全转换成 float。
    """
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def get_metric(obj, name, default=0.0):
    """
    安全获取 Ultralytics metric 属性。
    """
    try:
        value = getattr(obj, name)
        return safe_float(value)
    except Exception:
        return default


def calculate_f1(precision, recall):
    """
    F1 = 2PR / (P + R)
    """
    if precision is None or recall is None:
        return 0.0

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def get_class_names(model):
    """
    获取模型类别名称。
    """
    names = model.names

    if isinstance(names, dict):
        return names

    return {i: name for i, name in enumerate(names)}


# ============================================================
# 单模型评测
# ============================================================


def evaluate_model(
        model_path,
        data_yaml,
        imgsz=640,
        batch=16,
        device="0",
        workers=8,
        conf=0.001,
        iou=0.7,
        max_det=300,
        project="results",
):
    """
    对单个 YOLOv8-seg 模型进行验证。
    """

    model_path = Path(model_path)

    print()
    print("=" * 80)
    print(f"开始测试模型: {model_path.name}")
    print("=" * 80)

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model = YOLO(str(model_path))

    # --------------------------------------------------------
    # 模型类别
    # --------------------------------------------------------

    class_names = get_class_names(model)

    # --------------------------------------------------------
    # 运行 validation
    # --------------------------------------------------------

    start_time = time.time()

    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        conf=conf,
        iou=iou,
        max_det=max_det,
        # 不保存大量预测图片
        plots=False,
        project=project,
        name=model_path.stem,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - start_time

    # ========================================================
    # Mask metrics
    # ========================================================

    seg_metrics = results.seg

    mask_precision = get_metric(seg_metrics, "mp")
    mask_recall = get_metric(seg_metrics, "mr")

    mask_map50 = get_metric(seg_metrics, "map50")
    mask_map75 = get_metric(seg_metrics, "map75")
    mask_map5095 = get_metric(seg_metrics, "map")

    mask_f1 = calculate_f1(mask_precision, mask_recall)

    # ========================================================
    # Box metrics
    # ========================================================

    box_metrics = results.box

    box_precision = get_metric(box_metrics, "mp")
    box_recall = get_metric(box_metrics, "mr")

    box_map50 = get_metric(box_metrics, "map50")
    box_map75 = get_metric(box_metrics, "map75")
    box_map5095 = get_metric(box_metrics, "map")

    box_f1 = calculate_f1(box_precision, box_recall)

    # ========================================================
    # Speed
    # ========================================================

    speed = getattr(results, "speed", {})

    preprocess_ms = safe_float(speed.get("preprocess", 0))

    inference_ms = safe_float(speed.get("inference", 0))

    postprocess_ms = safe_float(speed.get("postprocess", 0))

    total_ms = preprocess_ms + inference_ms + postprocess_ms

    # ========================================================
    # 每类别 Mask metrics
    # ========================================================

    class_metrics = []

    # Ultralytics 的 summary() 可以直接得到逐类别指标
    try:

        summary = results.seg.summary(normalize=True, decimals=6)

        for item in summary:

            class_id = item.get("class")

            if class_id is None:
                continue

            class_id = int(class_id)

            class_name = class_names.get(class_id, str(class_id))

            class_metrics.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": safe_float(item.get("metrics/precision(M)", 0)),
                    "recall": safe_float(item.get("metrics/recall(M)", 0)),
                    "map50": safe_float(item.get("metrics/mAP50(M)", 0)),
                    "map5095": safe_float(item.get("metrics/mAP50-95(M)", 0)),
                }
            )

    except Exception as e:

        print(f"[WARNING] 获取逐类别 Mask 指标失败: {e}")

    # ========================================================
    # 打印结果
    # ========================================================

    print()
    print("-" * 80)
    print(f"模型: {model_path.name}")
    print("-" * 80)

    print(f"Mask Precision : {mask_precision:.4f}")

    print(f"Mask Recall    : {mask_recall:.4f}")

    print(f"Mask F1        : {mask_f1:.4f}")

    print(f"Mask mAP50     : {mask_map50:.4f}")

    print(f"Mask mAP75     : {mask_map75:.4f}")

    print(f"Mask mAP50-95  : {mask_map5095:.4f}")

    print()

    print(f"Box Precision  : {box_precision:.4f}")

    print(f"Box Recall     : {box_recall:.4f}")

    print(f"Box F1         : {box_f1:.4f}")

    print(f"Box mAP50      : {box_map50:.4f}")

    print(f"Box mAP50-95   : {box_map5095:.4f}")

    print()

    print(f"Preprocess     : {preprocess_ms:.2f} ms")

    print(f"Inference      : {inference_ms:.2f} ms")

    print(f"Postprocess    : {postprocess_ms:.2f} ms")

    print(f"Total          : {total_ms:.2f} ms")

    if total_ms > 0:

        fps = 1000.0 / total_ms

        print(f"FPS            : {fps:.2f}")

    else:

        fps = 0.0

    # ========================================================
    # 类别指标
    # ========================================================

    if class_metrics:

        print()
        print("-" * 80)
        print("Mask Class Metrics")
        print("-" * 80)

        print(
            f"{'Class':<20}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'mAP50':>12}"
            f"{'mAP50-95':>14}"
        )

        print("-" * 80)

        for item in class_metrics:
            print(
                f"{item['class_name']:<20}"
                f"{item['precision']:>12.4f}"
                f"{item['recall']:>12.4f}"
                f"{item['map50']:>12.4f}"
                f"{item['map5095']:>14.4f}"
            )

    # ========================================================
    # 返回结果
    # ========================================================

    result = {
        "model": model_path.name,
        "model_path": str(model_path),
        # Mask
        "mask_precision": mask_precision,
        "mask_recall": mask_recall,
        "mask_f1": mask_f1,
        "mask_map50": mask_map50,
        "mask_map75": mask_map75,
        "mask_map5095": mask_map5095,
        # Box
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": box_f1,
        "box_map50": box_map50,
        "box_map75": box_map75,
        "box_map5095": box_map5095,
        # Speed
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
        "fps": fps,
        # Benchmark time
        "elapsed_seconds": elapsed,
        # Per class
        "classes": class_metrics,
    }

    return result


# ============================================================
# 保存 CSV
# ============================================================


def save_csv(results, output_file):
    """
    保存总体模型对比结果。
    """

    if not results:
        return

    fields = [
        "model",
        "mask_precision",
        "mask_recall",
        "mask_f1",
        "mask_map50",
        "mask_map75",
        "mask_map5095",
        "box_precision",
        "box_recall",
        "box_f1",
        "box_map50",
        "box_map75",
        "box_map5095",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
        "fps",
        "elapsed_seconds",
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()

        for result in results:
            row = {key: result.get(key) for key in fields}

            writer.writerow(row)


# ============================================================
# 保存逐类别 CSV
# ============================================================


def save_class_csv(results, output_file):
    """
    保存每个模型的逐类别 Mask 指标。
    """

    rows = []

    for result in results:

        model_name = result["model"]

        for item in result["classes"]:
            rows.append(
                {
                    "model": model_name,
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "map50": item["map50"],
                    "map5095": item["map5095"],
                }
            )

    if not rows:
        return

    fields = [
        "model",
        "class_id",
        "class_name",
        "precision",
        "recall",
        "map50",
        "map5095",
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 保存 JSON
# ============================================================


def save_json(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# ============================================================
# 打印总对比表
# ============================================================


def print_comparison(results):
    print()
    print()
    print("=" * 120)
    print("YOLOv8-seg Model Benchmark")
    print("=" * 120)

    header = (
        f"{'Model':<28}"
        f"{'Mask P':>10}"
        f"{'Mask R':>10}"
        f"{'Mask F1':>10}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
        f"{'Box P':>10}"
        f"{'Box R':>10}"
        f"{'FPS':>10}"
    )

    print(header)

    print("-" * 120)

    for result in results:

        model_name = result["model"]

        if len(model_name) > 27:
            model_name = model_name[:24] + "..."

        print(
            f"{model_name:<28}"
            f"{result['mask_precision']:>10.4f}"
            f"{result['mask_recall']:>10.4f}"
            f"{result['mask_f1']:>10.4f}"
            f"{result['mask_map50']:>10.4f}"
            f"{result['mask_map5095']:>12.4f}"
            f"{result['box_precision']:>10.4f}"
            f"{result['box_recall']:>10.4f}"
            f"{result['fps']:>10.2f}"
        )

    print("=" * 120)


# ============================================================
# 获取模型
# ============================================================


def collect_models(model_paths, model_dir):
    models = []

    if model_paths:

        for path in model_paths:

            path = Path(path)

            if not path.exists():
                print(f"[WARNING] 模型不存在: {path}")

                continue

            models.append(path)

    if model_dir:

        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        models.extend(sorted(model_dir.glob("*.pt")))

    # 去重
    unique_models = []

    seen = set()

    for model in models:

        model = model.resolve()

        if model not in seen:
            seen.add(model)

            unique_models.append(model)

    return unique_models


# ============================================================
# main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-seg multi-model benchmark")

    parser.add_argument("--data", required=True, help="YOLO dataset yaml")

    parser.add_argument("--models", nargs="+", default=None, help="多个 YOLO .pt 模型")

    parser.add_argument("--model-dir", default=None, help="模型目录，自动测试目录下所有 .pt")

    parser.add_argument("--imgsz", type=int, default=640, help="验证图片尺寸")

    parser.add_argument("--batch", type=int, default=16, help="batch size")

    parser.add_argument("--device", default="0", help="CUDA device，例如 0 / 0,1 / cpu")

    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")

    parser.add_argument(
        "--conf", type=float, default=0.001, help="confidence threshold"
    )

    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")

    parser.add_argument("--max-det", type=int, default=300, help="最大检测数量")

    parser.add_argument("--output", default="results", help="结果输出目录")

    args = parser.parse_args()

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 获取模型
    # --------------------------------------------------------

    models = collect_models(args.models, args.model_dir)

    if not models:
        raise RuntimeError("没有找到任何 YOLO 模型")

    print()
    print("=" * 80)
    print("YOLOv8-seg Benchmark")
    print("=" * 80)

    print(f"Dataset : {args.data}")

    print(f"Models  : {len(models)}")

    print(f"ImageSz : {args.imgsz}")

    print(f"Batch   : {args.batch}")

    print(f"Device  : {args.device}")

    # --------------------------------------------------------
    # 测试所有模型
    # --------------------------------------------------------

    all_results = []

    for model_path in models:

        try:

            result = evaluate_model(
                model_path=model_path,
                data_yaml=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                project=str(output_dir / "ultralytics"),
            )

            all_results.append(result)

        except Exception as e:

            print()
            print(f"[ERROR] 模型测试失败: " f"{model_path}")

            print(f"错误信息: {e}")

    if not all_results:
        raise RuntimeError("所有模型均测试失败")

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    save_csv(all_results, output_dir / "benchmark_results.csv")

    save_class_csv(all_results, output_dir / "class_metrics.csv")

    save_json(all_results, output_dir / "benchmark_results.json")

    # --------------------------------------------------------
    # 总对比
    # --------------------------------------------------------

    print_comparison(all_results)

    print()
    print(f"总体结果:")

    print(f"  {output_dir / 'benchmark_results.csv'}")

    print(f"  {output_dir / 'class_metrics.csv'}")

    print(f"  {output_dir / 'benchmark_results.json'}")

    print()


if __name__ == "__main__":
    main()

"""


python Comparison_Model_Performance_Evaluation.py \
    --data /home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/datasets/val.yaml \
    --models \
        /home/chenkejing/Desktop/yolov8s_person_seg_v1_4/weights/best.pt \
        /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_3/weights/best.pt \
        /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_2/weights/best.pt \
        /home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt \
        /home/chenkejing/PycharmProjects/ultralytics/yolo11s-seg.pt \
    --imgsz 640 \
    --batch 16 \
    --conf 0.1 \
    --device 0
    
    
"""
