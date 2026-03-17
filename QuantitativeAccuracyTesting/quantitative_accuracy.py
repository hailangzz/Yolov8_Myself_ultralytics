import numpy as np
from rknn.api import RKNN
import cv2
import sys


# 计算均方误差 (MSE)
def calculate_mse(output_fp32, output_quantized):
    return np.mean((output_fp32 - output_quantized) ** 2)


# 读取图像并进行预处理
def read_image(image_path, size=(640, 640)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return None
    image = cv2.resize(image, size)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # 从 HWC 转为 CHW 格式
    image = np.expand_dims(image, axis=0)  # 增加 batch 维度 (1, C, H, W)
    return image


# 加载并初始化 RKNN 模型
def load_rknn_model(model_path):
    rknn_context = RKNN()

    # 加载模型
    ret = rknn_context.load_rknn(model_path)
    if ret != 0:
        print(f"Failed to load model {model_path}")
        return None

    # 设置目标设备，并初始化运行时环境
    target_device = 'rk3588'  # 根据硬件选择目标设备
    ret = rknn_context.init_runtime(target=target_device)
    if ret != 0:
        print("Failed to initialize runtime")
        return None

    return rknn_context


# 运行推理
def run_inference(model, image):
    outputs = model.inference(inputs=[image])
    if outputs is None:
        print("Inference failed, no output returned.")
        return None
    return outputs[0]  # 假设模型有一个输出，如果有多个输出，请根据需要修改索引


def main(fp32_model_path, quantized_model_path, image_path):
    # 读取图像
    image = read_image(image_path)
    if image is None:
        return

    # 加载 FP32 和量化后的模型
    fp32_model = load_rknn_model(fp32_model_path)
    if fp32_model is None:
        return

    quantized_model = load_rknn_model(quantized_model_path)
    if quantized_model is None:
        return

    # 运行推理
    print("Running inference on FP32 model...")
    output_fp32 = run_inference(fp32_model, image)
    if output_fp32 is None:
        print("FP32 model inference failed.")
        return

    print("Running inference on quantized model...")
    output_quantized = run_inference(quantized_model, image)
    if output_quantized is None:
        print("Quantized model inference failed.")
        return

    # 假设输出是分类任务的概率或回归任务的数值，转换为 numpy 数组
    output_fp32 = np.array(output_fp32)
    output_quantized = np.array(output_quantized)

    # 计算均方误差 (MSE)
    mse = calculate_mse(output_fp32, output_quantized)
    print(f"MSE between FP32 and Quantized model: {mse:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rknn_quantize_analysis.py <model_fp32> <model_quantized> <input_image>")
        sys.exit(1)

    fp32_model_path = sys.argv[1]
    quantized_model_path = sys.argv[2]
    image_path = sys.argv[3]

    main(fp32_model_path, quantized_model_path, image_path)

    #