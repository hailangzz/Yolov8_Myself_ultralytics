import Myself_Custom_model_structure.myself_model_struct as rk_head
import onnx
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import head
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True,
                        help="PyTorch yolov8 weights (.pt)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument("--input-shape", nargs="+", type=int,
                        default=[1, 3, 640, 640],
                        help="Static input shape")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    # 替换 Detect 的 forward（你的 RKNN 版本）
    setattr(head.Detect, "forward", rk_head.detect_forward)

    # 读取 YOLOv8 pt 模型
    model = YOLO(args.weights).model
    model = model.fuse().eval()

    device = args.device
    model.to(device)

    # 静态shape 输入
    fake_input = torch.randn(args.input_shape).to(device)

    # warmup 2 次
    with torch.no_grad():
        for _ in range(2):
            _ = model(fake_input)

    # 输出文件路径
    save_path = args.weights.replace(".pt", ".onnx")

    # 固定输出名（保持不变）
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

    print("Exporting ONNX (static graph)...")

    try:
        # ----------- 🚀 静态图 ONNX 导出（无 dynamic_axes） -----------
        torch.onnx.export(
            model,
            fake_input,
            save_path,                 # 直接写入文件
            opset_version=args.opset,
            input_names=["images"],
            output_names=output_names,
            do_constant_folding=True,  # 静态优化
            dynamic_axes=None,         # ❗静态图关键点：关闭动态 shape
        )
        # --------------------------------------------------------------

        # 检查模型
        model_onnx = onnx.load(save_path)
        onnx.checker.check_model(model_onnx)
        print("ONNX model check passed ✔")

        # 可选：简化
        if args.sim:
            try:
                import onnxsim
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check
                onnx.save(model_onnx, save_path)
                print("ONNX simplified ✔")
            except Exception as e:
                print(f"ONNX simplifier failed: {e}")

        print(f"ONNX export success (Static Graph), saved at:\n  {save_path}")

    except Exception as e:
        print(f"Export Error: {e}")


if __name__ == "__main__":
    main(parse_args())
