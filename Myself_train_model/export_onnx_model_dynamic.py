import argparse
from io import BytesIO

import onnx
import torch

import Myself_Custom_model_structure.myself_model_struct as rk_head
from ultralytics import YOLO
from ultralytics.nn.modules import head

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True, help="PyTorch yolov8 weights")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 640, 640],
        help="Model input shape only for api builder",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Export ONNX device")
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    # 如果你有自定义 Detect forward，例如 RKNN 修改
    setattr(head.Detect, "forward", rk_head.detect_forward)

    # 加载 YOLOv8 PyTorch 模型
    model = YOLO(args.weights).model  # 获取 PyTorch model
    model = model.fuse().eval()  # Fuse Conv + BN

    # 将模型移动到指定设备
    device = args.device if hasattr(args, "device") else "cpu"
    model.to(device)

    # 构造假输入 (batch, channels, height, width)
    fake_input = torch.randn(args.input_shape).to(device)

    # warmup，跑两次前向
    with torch.no_grad():
        for _ in range(2):
            _ = model(fake_input)

    # ONNX 保存路径
    save_path = args.weights.replace(".pt", ".onnx")

    # 输出名列表（根据 YOLOv8 输出）
    output_names = [
        "yolov8_detect_output0_box",
        "yolov8_detect_output0_class",
        "yolov8_detect_output0_class_sum",
        "yolov8_detect_output1_box",
        "yolov8_detect_output1_class",
        "yolov8_detect_output1_class_sum",
        "yolov8_detect_output2_box",
        "yolov8_detect_output2_class",
        "yolov8_detect_output2_class_sum",
    ]

    # 导出 ONNX
    try:
        with BytesIO() as f:
            torch.onnx.export(
                model,
                fake_input,
                f,
                opset_version=getattr(args, "opset", 17),
                input_names=["images"],
                output_names=output_names,
                dynamic_axes={
                    "images": {0: "batch"},  # batch 可变
                    **{name: {0: "batch"} for name in output_names},
                },
            )
            f.seek(0)
            onnx_model = onnx.load(f)

        # 检查 ONNX 模型
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed")

        # 可选简化
        if getattr(args, "sim", False):
            try:
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, "ONNX simplifier check failed"
                print("ONNX model simplified")
            except Exception as e:
                print(f"ONNX simplifier failure: {e}")

        # 保存 ONNX 文件
        onnx.save(onnx_model, save_path)
        print(f"ONNX export success, saved as {save_path}")

    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    main(parse_args())
