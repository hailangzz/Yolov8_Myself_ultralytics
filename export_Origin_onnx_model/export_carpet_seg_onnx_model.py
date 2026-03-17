from ultralytics import YOLO

if __name__ == "__main__":
    # 1️⃣ 加载训练好的 YOLOv8-seg 模型
    model = YOLO("/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_1/weights/best.pt")

    # 2️⃣ 导出模型
    # 可选格式：onnx, torchscript, engine (TensorRT), coreml
    # 这里举例导出 ONNX
    model.export(format="onnx", opset=17, simplify=True)

    # 如果想同时导出 TorchScript
    # model.export(format="torchscript")

    # 如果想导出 TensorRT
    # model.export(format="engine")

    print("✅ YOLOv8-seg 模型导出完成！")
