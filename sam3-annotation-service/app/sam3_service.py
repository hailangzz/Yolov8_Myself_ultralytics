from ultralytics.models.sam import SAM3SemanticPredictor


class SAM3Service:
    def __init__(self, model_path, conf=0.25, quantize=16, device="cuda"):

        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "quantize": quantize,
            "save": False,
            "device": device,
        }

        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def predict(self, image_path, prompts):

        # 设置图片
        self.predictor.set_image(image_path)

        # 文本提示
        results = self.predictor(text=prompts)

        return results
