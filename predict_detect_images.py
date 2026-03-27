import argparse
import os

import cv2

from ultralytics import YOLO


def run_inference(model_path, imgs_dir, save_dir):
    # 自动选择设备
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    # 加载模型到指定设备
    model = YOLO(model_path)
    model.to(device)

    # 结果保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 支持的图片格式
    exts = (".jpg", ".jpeg", ".png", ".bmp")

    # 遍历目录下所有图片
    for img_name in os.listdir(imgs_dir):
        if not img_name.lower().endswith(exts):
            continue

        img_path = os.path.join(imgs_dir, img_name)
        print(f"Processing {img_path}")

        # 推理
        results = model(img_path, conf=0.55)[0]
        # results = model(img_path, conf=0.35)[0]
        # results = model(img_path, conf=0.0005)[0]

        # 读取原图
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (640, 640))

        # 绘制检测框
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls]} {conf:.2f}"

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存结果
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)
        print(f"> Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained YOLO model")
    parser.add_argument("--imgs_dir", type=str, required=True, help="Directory containing the images to infer")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the inference results")
    return parser.parse_args()


if __name__ == "__main__":
    # model_path = "/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt"         # 修改为你的模型路径
    # imgs_dir = "/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test"             # 输入图片目录
    # save_dir = "./results/carpet"            # 结果输出目录
    # run_inference(model_path, imgs_dir, save_dir)

    args = parse_args()
    run_inference(args.model_path, args.imgs_dir, args.save_dir)
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test --save_dir ./results/carpet
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_v3/weights/best.pt --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test --save_dir ./results/carpet

    # 误检样本测试
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_v3/weights/best.pt --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_negative_test --save_dir ./results/carpet

    # 测试手势识别 yolov8_focus_sa_v2
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_hand_exp/yolov8_focus_sa_v2/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir ./results/hand_model_sa_v2
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_hand_exp/yolov8_focus_v3/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir ./results/hand_model_v3
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_hand_exp/yolov8_focus_v7/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir ./results/hand_model_v7

    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v2/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir ./results/hand_focus_sa_v2
    # 3月11日
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v2/weights/best.pt --imgs_dir /home/chenkejing/Desktop/rgb_images --save_dir ./results/hand_focus_sa_v2_rgb

    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v3_5/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir ./results/hand_yolov8_focus_sa_v3_5

    # 测试地毯检测真实样本
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_7/weights/best.pt --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_real_image --save_dir ./results/carpet
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v3_3/weights/best.pt --imgs_dir /home/chenkejing/Desktop/capture_images --save_dir ./results/hand_model_focus_sa_v3

    # 液体检测
    # 3月18日
    # python predict_detect_images.py --model_path /home/chenkejing/Desktop/yolov8s_Liquad_det_v1_5/weights/best.pt --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/liquad_image_test --save_dir ./results/liquad_model_focus_v3
