import torch
from ultralytics import YOLO
from rknn.api import RKNN
import argparse
import os

def export_yolov8_to_rknn(model_path , onnx_path, rknn_path, imgsz=640, do_quant=False, calib_data=None):
    # 1️⃣ 加载 YOLOv8 模型
    model = YOLO(model_path)

    # 2️⃣ 创建 dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz)

    # 3️⃣ 导出 ONNX（裁掉后处理）
    # 直接用 model.model(x) 输出 tensor，不走 NMS
    torch.onnx.export(
        model.model,
        dummy_input,
        onnx_path,
        opset_version=16,
        input_names=["images"],
        output_names=["preds"],
        dynamic_axes={"images": {0: "batch"}, "preds": {0: "batch"}}
    )
    print(f"[INFO] ONNX model exported to {onnx_path}")

    # 4️⃣ 转 RKNN
    rknn = RKNN()
    rknn.config(
        mean_values=[[0, 0, 0]],  # 可根据训练预处理修改
        std_values=[[255, 255, 255]],
        target_platform="rk3588",  # 或 rk3588 / rk3399pro
        reorder_channel="0 1 2",  # RGB->RGB
    )
    rknn.load_onnx(model=onnx_path)

    # 量化可选
    rknn.build(do_quantization=do_quant, dataset=calib_data)
    rknn.export_rknn(rknn_path)
    print(f"[INFO] RKNN model exported to {rknn_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 PT to RKNN (no postprocessing)")
    parser.add_argument("--pt", type=str, required=True, default = r"/home/chenkejing/PycharmProjects/ultralytics/runs/detect/train3/weights/best.pt",help="Path to YOLOv8 .pt model")
    parser.add_argument("--onnx", type=str, default="model_no_post.onnx", help="Output ONNX path")
    parser.add_argument("--rknn", type=str, default="model_no_post.rknn", help="Output RKNN path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--quant", action="store_true", help="Enable quantization")
    parser.add_argument("--calib", type=str, default=None, help="Path to calibration dataset for quantization")
    args = parser.parse_args()

    export_yolov8_to_rknn(args.pt, args.onnx, args.rknn, imgsz=args.imgsz, do_quant=args.quant, calib_data=args.calib)