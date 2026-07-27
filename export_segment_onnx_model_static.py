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
    head.Segment.forward = rk_head.segment_forward
    head.Detect.forward = rk_head.detect_forward
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    model.to(args.device)

    # ======== EXPORT PATCH START ========
    from ultralytics.nn.modules.block import C2f
    from ultralytics.nn.modules.head import Segment

    for m in model.modules():
        # 1. Detect / Segment export 模式（必须）
        if isinstance(m, Segment):
            m.export = True
            m.dynamic = False
            m.training = False

        # 2. Focus export 模式（关键）
        if m.__class__.__name__ == "Focus":
            m.export = True

        # 3. C2f ONNX safe（避免 split graph 问题）
        if isinstance(m, C2f) and hasattr(m, "forward_split"):
            m.forward = m.forward_split

    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace(".pt", ".onnx")
    output_names = [
        "yolov8_output0_box",
        "yolov8_output0_class",
        "yolov8_output0_class_sum",
        "yolov8_output0_mask",
        "yolov8_output1_box",
        "yolov8_output1_class",
        "yolov8_output1_class_sum",
        "yolov8_output1_mask",
        "yolov8_output2_box",
        "yolov8_output2_class",
        "yolov8_output2_class_sum",
        "yolov8_output2_mask",
        "yolov8_proto",
    ]
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=["images"],
            output_names=output_names,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), save_path)
    print(f"ONNX export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())
