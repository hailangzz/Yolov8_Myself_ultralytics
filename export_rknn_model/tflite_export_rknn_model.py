from rknn.api import RKNN

DATASET_PATH = (
    "/home/chenkejing/PycharmProjects/EMdoorTotalDetect/rk3588-convert-to-rknn/wire_quant_data/wire_quant.txt"
)
origin_model_path = r"/home/chenkejing/EMdoor_TotalProgram/MediapipeDemo/models/palm_detection_full.tflite"
Target_Platform = "rk3568"

# 创建RKNN对象
rknn = RKNN(verbose=True)
# rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform='rk3588')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform=Target_Platform)

# 加载模型
ret = rknn.load_tflite(model=origin_model_path)
if ret != 0:
    print("Load RKNN model failed!")
    exit(ret)

# Build model
print("--> Building model")
ret = rknn.build(do_quantization=False)  # 非量化
# ret = rknn.build(do_quantization=True,dataset=DATASET_PATH)  # 量化
if ret != 0:
    print("Build model failed!")
    exit(ret)
print("done")

# Export rknn model
print("--> Export rknn model")

output_model_name = "./palm_detection_full_" + Target_Platform + ".rknn"
ret = rknn.export_rknn(output_model_name)
if ret != 0:
    print("Export rknn model failed!")
    exit(ret)
print("done")
