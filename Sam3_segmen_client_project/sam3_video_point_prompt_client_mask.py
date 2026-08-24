#!/usr/bin/env python3

import base64
import os
import time

import numpy as np
import requests
from PIL import Image


class SAM3VideoClient:
    def __init__(self, server="http://127.0.0.1:9000", model="segment_anything_3_video"):

        self.server = server.rstrip("/")
        self.model = model

    def encode_image(self, image_path):

        with open(image_path, "rb") as f:
            img = f.read()

        return base64.b64encode(img).decode()

    def load_images(self, image_dir):

        images = []

        for name in sorted(os.listdir(image_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(image_dir, name))

        print(f"Found {len(images)} images")

        return images

    ########################################
    # 初始化多图片序列
    ########################################

    def init_sequence(self, image_paths):

        print("Encoding images...")

        frames = []

        for img in image_paths:
            frames.append(self.encode_image(img))

        payload = {"model": self.model, "frames": frames, "start_frame_index": 0}

        r = requests.post(self.server + "/v1/video/init", json=payload, timeout=300)

        r.raise_for_status()

        result = r.json()

        print(result)

        session_id = result["data"]["session_id"]

        print("session:", session_id)

        return session_id

    ########################################
    # 点提示
    ########################################

    def prompt_point(self, session_id, x, y, frame_index=0):

        payload = {
            "session_id": session_id,
            "model": self.model,
            "frame_index": frame_index,
            "points": [[x, y]],
            "point_labels": [1],
            "obj_id": 1,
        }

        r = requests.post(self.server + "/v1/video/prompt", json=payload, timeout=300)

        result = r.json()

        print("Prompt result:", result)

        return result

    ########################################
    # 开始传播
    ########################################

    def propagate(self, session_id, num_frames):

        payload = {"session_id": session_id, "model": self.model, "start_frame": 0, "end_frame": num_frames - 1}

        r = requests.post(self.server + "/v1/video/propagate", json=payload, timeout=300)

        result = r.json()

        print("Propagate:", result)

        return result["data"]["task_id"]

    ########################################
    # 等待结果
    ########################################

    def wait(self, task_id):

        while True:
            r = requests.get(self.server + f"/v1/video/status/{task_id}", timeout=60)

            result = r.json()

            print("Status:", result)

            data = result.get("data", {})

            status = data.get("status")

            if status == "completed":
                return data

            if status in ["failed", "cancelled"]:
                raise RuntimeError(result)

            time.sleep(2)

    ########################################
    # 保存mask
    ########################################

    def save_mask_png(self, mask_data, save_path):
        """支持: 1. base64 png 2. base64 numpy mask.
        """
        try:
            raw = base64.b64decode(mask_data)

            img = Image.open(io.BytesIO(raw))

            img.save(save_path)

        except Exception:
            mask = np.frombuffer(base64.b64decode(mask_data), dtype=np.uint8)

            img = Image.fromarray(mask)

            img.save(save_path)

    ########################################
    # 解析并保存结果
    ########################################

    def save_results(self, result, output_dir="./masks"):

        os.makedirs(output_dir, exist_ok=True)

        print("Saving masks...")

        print("Result:", result)

        #
        # 情况1:
        #
        # results=[
        # {
        # frame_index:0,
        # mask:"base64"
        # }
        # ]
        #

        if "results" in result:
            results = result["results"]

            if isinstance(results, list):
                for item in results:
                    frame_id = item.get("frame_index", 0)

                    mask = item.get("mask")

                    if mask:
                        path = os.path.join(output_dir, f"{frame_id:06d}.png")

                        self.save_mask_png(mask, path)

        print("Masks saved:", output_dir)


#############################################
# 测试
#############################################

if __name__ == "__main__":
    client = SAM3VideoClient(server="http://172.16.50.229:9000")

    image_dir = "./images"

    images = client.load_images(image_dir)

    session_id = client.init_sequence(images)

    client.prompt_point(session_id, x=300, y=200)

    task_id = client.propagate(session_id, len(images))

    result = client.wait(task_id)

    print("FINAL RESULT")

    print(result)

    client.save_results(result, "./masks")
