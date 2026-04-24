import argparse
import os

import cv2
import numpy as np
import torch

from ultralytics import YOLO


def run_segmentation(model_path, imgs_dir, save_dir, conf_thres=0.55):
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    model = YOLO(model_path)
    model.to(device)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")

    # 所有类别统一颜色（绿色）
    num_classes = len(model.names)
    # class_colors = {i: np.array([0, 255, 0]) for i in range(num_classes)}
    class_colors = {i: (0, 255, 0) for i in range(num_classes)}

    for img_name in os.listdir(imgs_dir):
        if not img_name.lower().endswith(exts):
            continue

        img_path = os.path.join(imgs_dir, img_name)
        print(f"[INFO] Processing {img_path}")

        # 推理
        results = model(
            img_path,
            conf=conf_thres,
            task="segment",
            imgsz=640,
            # imgsz=960
            # imgsz=1280
        )[0]

        # 读取原图
        img = cv2.imread(img_path)
        overlay = img.copy()

        # ========================
        # ✅ 正确绘制 mask（关键修复）
        # ========================
        if results.masks is not None:
            segments = results.masks.xy  # 已经映射回原图坐标
            classes = results.boxes.cls.cpu().numpy()

            for i, segment in enumerate(segments):
                pts = np.array(segment, dtype=np.int32)

                cls = int(classes[i])
                color = class_colors[cls]

                cv2.fillPoly(overlay, [pts], color)

        # 半透明叠加
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        # ========================
        # 绘制检测框
        # ========================
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls)
                conf = float(box.conf)

                label = f"{model.names[cls]} {conf:.2f}"

                x1, y1, x2, y2 = map(int, xyxy)

                color = tuple(int(c) for c in class_colors[cls])

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存结果
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)

        print(f"[INFO] Saved: {save_path}")


def parse_args():

    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Inference Script")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained YOLOv8 segmentation model (.pt)"
    )

    parser.add_argument("--imgs_dir", type=str, required=True, help="Directory containing images to infer")

    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the inference results")

    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold for detection")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_segmentation(args.model_path, args.imgs_dir, args.save_dir, args.conf)

    # 示例运行命令：
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_7/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_real_image  --save_dir ./results/carpet  --conf 0.55

    # 地毯检测
    # 0416
    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_10/weights/best.pt  --imgs_dir /home/chenkejing/Desktop/images  --save_dir ./results/carpet  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage1_head_only/weights/best.pt  --imgs_dir /home/chenkejing/database/object_camera_coordinates_image/carpet_detect/date0416/images  --save_dir ./results/carpet  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_finetune_model/runs/carpet_seg_finetune/stage2_full_finetune/weights/best.pt  --imgs_dir /home/chenkejing/database/object_camera_coordinates_image/carpet_detect/date0416/images  --save_dir ./results/carpet  --conf 0.55

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

    # 0401液体检测
    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_liquid_seg_v1_rect_boxgain/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image  --save_dir ./results/liquid  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_liquid_exp/yolov8s_liquid_seg_v1_rect_boxgain/weights/best.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image  --save_dir ./results/liquid  --conf 0.55

    # python predict_seg_images.py  --model_path /home/chenkejing/Desktop/yolov8s_liquid_seg_v1_rect_boxgain7/weights/last.pt  --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_real_image  --save_dir ./results/liquid  --conf 0.55
