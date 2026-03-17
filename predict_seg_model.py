# yolov8_seg_batch_inference.py

from ultralytics import YOLO
import cv2
import os
import glob

# -----------------------------
# 配置
# -----------------------------
model_path = '/home/chenkejing/PycharmProjects/ultralytics/runs/my_wire_seg_exp/yolov8s_wire_seg_v1_2/weights/best.pt'  # 模型路径
input_dir = '/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test'  # 输入图像目录
output_dir = './results/wire'  # 保存目录
img_exts = ['*.jpg', '*.png', '*.jpeg']  # 支持的图片格式
img_size = 640  # 推理输入尺寸
conf_thresh = 0.25  # 置信度阈值
iou_thresh = 0.45  # NMS 阈值
device = '0'  # GPU 0，如果没有 GPU 改成 'cpu'

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 加载模型
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# 获取所有图像路径
# -----------------------------
image_paths = []
for ext in img_exts:
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

if not image_paths:
    print("[WARN] 输入目录没有找到图片！")
    exit(0)

print(f"[INFO] 找到 {len(image_paths)} 张图片进行推理。")

# -----------------------------
# 批量推理
# -----------------------------
for img_path in image_paths:
    results = model.predict(
        source=img_path,
        imgsz=img_size,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        save=False
    )

    for i, result in enumerate(results):
        # result.plot() 会返回绘制了 mask 和 bbox 的 numpy 图像
        output_image = result.plot()

        # 保存为 PNG
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_mask.png')
        cv2.imwrite(output_path, output_image)
        print(f"[INFO] {img_path} -> {output_path}")

print("[INFO] 所有图片推理完成！")