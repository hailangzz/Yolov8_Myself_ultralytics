# -*- coding: utf-8 -*-

"""
YOLO11-seg inference
Only export person segmentation
Output YOLO-seg labels
class id = 0
"""

from pathlib import Path
import cv2
import numpy as np

from ultralytics import YOLO


# ==========================
# mask 转 YOLO polygon
# ==========================


def mask_to_yolo_polygon(mask):
    """
    mask:
        H x W
        uint8 0/1


    return:
        [
          x1,y1,x2,y2...
        ]

    normalized
    """

    h, w = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 最大轮廓

    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.002 * cv2.arcLength(contour, True)

    contour = cv2.approxPolyDP(contour, epsilon, True)

    points = []

    for point in contour:
        x, y = point[0]

        points.append(x / w)

        points.append(y / h)

    return points


# ==========================
# 推理
# ==========================


def inference():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    print(f"Images: {len(images)}")

    for img_path in images:

        print(f"\nProcessing {img_path.name}")

        results = model(
            str(img_path),
            conf=CONF_THRESHOLD,
            verbose=False
        )

        result = results[0]

        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        # 默认创建空标签
        labels = []

        if result.masks is not None:

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            for cls_id, mask in zip(classes, masks):

                # 只保留person
                if cls_id != 0:
                    continue

                polygon = mask_to_yolo_polygon(mask)

                if polygon is None:
                    continue

                line = (
                        "0 "
                        +
                        " ".join(
                            [
                                f"{x:.6f}"
                                for x in polygon
                            ]
                        )
                )

                labels.append(line)

        # ==========================
        # 无论是否检测到目标
        # 都生成txt
        # ==========================

        with open(
                label_path,
                "w",
                encoding="utf-8"
        ) as f:

            if labels:
                f.write(
                    "\n".join(labels)
                )

        if labels:

            print(
                f"Saved {label_path}"
            )

        else:

            print(
                f"Empty label: {label_path}"
            )


# ==========================
# 配置
# ==========================

# IMAGE_DIR = Path("/home/chenkejing/Downloads/WireSegmentProject/person_images_yolov11-seg")
#
# LABEL_DIR = Path(
#     "/home/chenkejing/Downloads/WireSegmentProject/person_detect_yolov11_auto_labels")


IMAGE_DIR = Path(
    "/home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1_output_empty_only/images")
#
LABEL_DIR = Path(
    "/home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1_output_empty_only/person_labels")

MODEL_PATH = "yolo11m-seg.pt"

CONF_THRESHOLD = 0.35

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if __name__ == "__main__":
    inference()
