'''
说明：此代码的作用是，输入一个图片文件夹，还是用训练好的检测模型，检测到样本集里的所有目标样本；
    然后将带目标的样本，保存到给定目录下；

作用：抽取有效目标样本，扩充训练集

'''

import os
from ultralytics import YOLO
import cv2
import argparse


def run_inference(model_path, imgs_dir, save_dir):
    # 加载模型
    model = YOLO(model_path)

    # 结果保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 支持的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 遍历目录下所有图片
    for img_name in os.listdir(imgs_dir):
        if not img_name.lower().endswith(exts):
            continue

        img_path = os.path.join(imgs_dir, img_name)
        print(f"Processing {img_path}")

        # 推理
        results = model(img_path, conf=0.55)[0]

        # 读取原图
        img = cv2.imread(img_path)

        # # 绘制检测框
        # for box in results.boxes:
        #     xyxy = box.xyxy[0].cpu().numpy()
        #     cls = int(box.cls)
        #     conf = float(box.conf)
        #     label = f"{model.names[cls]} {conf:.2f}"
        #
        #     x1, y1, x2, y2 = map(int, xyxy)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(img, label, (x1, y1 - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 2)

        if len(results.boxes)>0:
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
    # python predict_detect_images.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_v3/weights/best.pt --imgs_dir /home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test --save_dir ./results/carpet
    # python /home/chenkejing/PycharmProjects/ultralytics/Myself_inference_model/predict_images_save_exist_object_image.py --model_path /home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_v3/weights/best.pt --imgs_dir /home/chenkejing/Downloads/homeobjects-3K/images/train --save_dir /home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/origin_public_carpet_database/add_images_homeobjects_3k


    """
    python3 /home/chenkejing/PycharmProjects/ultralytics/Myself_inference_model/predict_images_save_exist_object_image.py --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v2/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir /home/chenkejing/PycharmProjects/ultralytics/results/hand_focus_sa_v2
    
    
    
    #0310日期：
    # python3 /home/chenkejing/PycharmProjects/ultralytics/Myself_inference_model/predict_images_save_exist_object_image.py --model_path /home/chenkejing/PycharmProjects/ultralytics/runs/my_hand_exp/yolov8_focus_sa_v2/weights/best.pt --imgs_dir /home/chenkejing/Desktop/hand_detect --save_dir /home/chenkejing/database/Negativew_Example_Dataset/hand/Negative_hand_batch_0310_database/images
    
    
    """