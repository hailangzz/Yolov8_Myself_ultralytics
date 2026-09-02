import os
from ultralytics import YOLO
import cv2
import argparse
import torch
import numpy as np


def run_segmentation(
        model_path,
        imgs_dir,
        save_dir,
        conf_thres=0.55,
        target_classes=None
):
    # ==========================
    # 自动选择设备
    # ==========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================
    # 加载模型
    # ==========================
    model = YOLO(model_path)
    model.to(device)

    print(f"[INFO] Model classes: {model.names}")

    # ==========================
    # 解析目标类别
    # ==========================
    if target_classes is not None:

        target_class_ids = []

        for class_name in target_classes:

            # 根据类别名称查找 class id
            matched = [
                cls_id
                for cls_id, name in model.names.items()
                if name == class_name
            ]

            if not matched:
                print(
                    f"[WARNING] Class '{class_name}' "
                    f"not found in model.names"
                )
                continue

            target_class_ids.append(matched[0])

        if not target_class_ids:
            raise ValueError(
                "No valid target classes were found."
            )

        print(
            f"[INFO] Only draw classes: "
            f"{target_classes}"
        )

        print(
            f"[INFO] Target class IDs: "
            f"{target_class_ids}"
        )

    else:
        # None 表示绘制所有类别
        target_class_ids = list(model.names.keys())

        print("[INFO] Draw all classes")

    # ==========================
    # 创建保存目录
    # ==========================
    os.makedirs(save_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # ==========================
    # 所有类别统一颜色（绿色）
    # ==========================
    num_classes = len(model.names)

    class_colors = {
        i: (0, 255, 0)
        for i in range(num_classes)
    }

    # ==========================
    # 遍历图片
    # ==========================
    for img_name in os.listdir(imgs_dir):

        if not img_name.lower().endswith(exts):
            continue

        img_path = os.path.join(imgs_dir, img_name)

        print(f"[INFO] Processing {img_path}")

        # ==========================
        # 推理
        # ==========================
        results = model(
            img_path,
            conf=conf_thres,
            task='segment',
            imgsz=640
        )[0]

        # ==========================
        # 读取原图
        # ==========================
        img = cv2.imread(img_path)

        if img is None:
            print(
                f"[WARNING] Failed to read image: "
                f"{img_path}"
            )
            continue

        overlay = img.copy()

        # ==========================
        # 绘制 Mask
        # ==========================
        if results.masks is not None:

            segments = results.masks.xy
            classes = results.boxes.cls.cpu().numpy()

            for i, segment in enumerate(segments):

                cls = int(classes[i])

                # ==========================
                # ⭐ 类别筛选
                # ==========================
                if cls not in target_class_ids:
                    continue

                pts = np.array(
                    segment,
                    dtype=np.int32
                )

                color = class_colors[cls]

                cv2.fillPoly(
                    overlay,
                    [pts],
                    color
                )

        # ==========================
        # 半透明叠加
        # ==========================
        img = cv2.addWeighted(
            overlay,
            0.7,
            img,
            0.3,
            0
        )

        # ==========================
        # 绘制检测框
        # ==========================
        if results.boxes is not None:

            for box in results.boxes:

                cls = int(box.cls)

                # ==========================
                # ⭐ 类别筛选
                # ==========================
                if cls not in target_class_ids:
                    continue

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                conf = float(box.conf)

                label = (
                    f"{model.names[cls]} "
                    f"{conf:.2f}"
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                color = tuple(
                    int(c)
                    for c in class_colors[cls]
                )

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # ==========================
        # 保存结果
        # ==========================
        save_path = os.path.join(
            save_dir,
            img_name
        )

        cv2.imwrite(
            save_path,
            img
        )

        print(
            f"[INFO] Saved: {save_path}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "YOLOv8 Segmentation "
            "Inference Script"
        )
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the trained "
            "YOLO segmentation model (.pt)"
        )
    )

    parser.add_argument(
        "--imgs_dir",
        type=str,
        required=True,
        help=(
            "Directory containing "
            "images to infer"
        )
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help=(
            "Directory to save "
            "the inference results"
        )
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.55,
        help=(
            "Confidence threshold "
            "for detection"
        )
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=(
            "Target classes to draw. "
            "Example: --classes carpet wire"
        )
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_segmentation(
        model_path=args.model_path,
        imgs_dir=args.imgs_dir,
        save_dir=args.save_dir,
        conf_thres=args.conf,
        target_classes=args.classes
    )

    # 示例运行命令：
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_7/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_real_image_plus  --save_dir ./results/carpet  --conf 0.55

    # 地毯检测
    # 0416
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_10/weights/best.pt  --imgs_dir /home/chenkejing/Desktop/images  --save_dir ./results/carpet  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage1_head_only/weights/best.pt  --imgs_dir /home/chenkejing/database/object_camera_coordinates_image/carpet_detect/date0416/images  --save_dir ./results/carpet  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune/weights/best.pt  --imgs_dir /home/chenkejing/database/object_camera_coordinates_image/carpet_detect/date0416/images  --save_dir ./results/carpet  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune/weights/best.pt  --imgs_dir /data/database/AITotal_Segment_ValDatabase/public_real_camera_images_0422_carpet_val_batch1  --save_dir ./results/carpet  --conf 0.55

    # 4月17
    # python predict_seg_images.py  --model_path  /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune2/weights/best.pt --imgs_dir /home/chenkejing/Downloads/images/images  --save_dir ./results/carpet  --conf 0.55

    # python predict_seg_images.py  --model_path  /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune2/weights/best.pt --imgs_dir /home/chenkejing/Downloads/images  --save_dir ./results/carpet  --conf 0.55

    # 5月11日：
    # python predict_seg_images.py  --model_path  /home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_5/weights/best.pt --imgs_dir /data/database/AITotal_Real_Customer_Database/Real_Carpet_Customer_Database/date0508/images  --save_dir ./results/carpet  --conf 0.55
    # python predict_seg_images.py  --model_path  /home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_5/weights/best.pt --imgs_dir /data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0519/WireSegmentProject/images  --save_dir ./results/carpet  --conf 0.55

    # 6月15日：
    # python predict_seg_images.py  --model_path  /home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_6/weights/best.pt --imgs_dir /data/database/AITotal_Real_Customer_Database/Real_Carpet_Customer_Database/date0612/images  --save_dir ./results/carpet  --conf 0.55

    # 线材检测
    # 0316线材检测
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_wire_seg_exp/yolov8s_wire_seg_v1_2/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # 0327线材检测
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_v1_5/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # 0331线材检测
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_v1_rect_boxgain4/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_wire_seg_exp/yolov8s_wire_seg_v1_5/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_v2_/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55
    # 0407
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_v2_4/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_finetune_stage1/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_finetune_stage22/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # 0427
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/yolov8s_wire_seg_finetune_stage1/stage2_full_finetune/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # 0518
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_v2_6/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55
    # 0519
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_finetune_stage14/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_wire_seg_exp/yolov8s_wire_seg_finetune_stage25/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_wire_seg_exp/yolov8s_wire_seg_finetune_stage25/weights/best.pt  --imgs_dir /data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0514/WireSegmentProject/spatial_location_val_images/null_target  --save_dir ./results/wire  --conf 0.55

    # 6月8号
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_wire_seg_finetune_stage16/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test  --save_dir ./results/wire  --conf 0.55

    # 0401液体检测
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_liquid_seg_v1_rect_boxgain/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image  --save_dir ./results/liquid  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_exp/yolov8s_liquid_seg_v1_rect_boxgain/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image  --save_dir ./results/liquid  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_liquid_seg_v1_rect_boxgain7/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image  --save_dir ./results/liquid  --conf 0.55

    # 5月8日
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_rect_boxgain8/weights/best.pt  --imgs_dir /data/database/AITotal_Real_Customer_Database/Real_Liquid_Customer_Database/date0511/images --save_dir ./results/liquid  --conf 0.55
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_rect_boxgain8/weights/best.pt  --imgs_dir /data/database/coco2017/test2017 --save_dir ./results/liquid  --conf 0.55

    # 6月1日
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_rect_boxgain9/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image --save_dir ./results/liquid  --conf 0.55

    # 6月29日
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_rect_boxgain11/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image --save_dir ./results/liquid  --conf 0.55
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_seg_exp/yolov8s_liquid_seg_v1_rect_boxgain13/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image --save_dir ./results/liquid  --conf 0.55

    # 7月1日塑料袋检测
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_plasticbag_seg_exp/yolov8s_plasticbag_seg_v1_4/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/plastic_bag_real_image  --save_dir ./results/plastic_bag  --conf 0.55

    # 行人检测
    # 8月24日
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_2/weights/best.pt  --imgs_dir /home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1/images  --save_dir ./results/person_v1  --conf 0.3

    # 8月31日
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_3/weights/best.pt  --imgs_dir /home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1/images  --save_dir ./results/person  --conf 0.3

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_2/weights/best.pt  --imgs_dir /data/database/jrdb_yolo_random_val/images/val  --save_dir ./results/person_v2  --conf 0.3
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_3/weights/best.pt  --imgs_dir /data/database/jrdb_yolo_random_val/images/val  --save_dir ./results/person_v3  --conf 0.3

    # python predict_seg_images.py   --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_3/weights/best.pt  --imgs_dir /data/database/jrdb_yolo_random_val/images/val  --save_dir ./results/person_v3   --conf 0.3  --classes person

    # 官方模型
    # python predict_seg_images_target_classify.py   --model_path /home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt  --imgs_dir /data/database/AITotal_SegmentDatabase/personDatabaseSegment/date_20260825/images/val  --save_dir ./results/person_base   --conf 0.3  --classes person

    # python predict_seg_images_target_classify.py   --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_2/weights/best.pt  --imgs_dir /data/database/AITotal_SegmentDatabase/personDatabaseSegment/date_20260825/images/val  --save_dir ./results/person_v2   --conf 0.3  --classes person

    # python predict_seg_images_target_classify.py   --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_person_seg_exp/yolov8s_person_seg_v1_3/weights/best.pt  --imgs_dir /data/database/AITotal_SegmentDatabase/personDatabaseSegment/date_20260825/images/val  --save_dir ./results/person_v3   --conf 0.3  --classes person
