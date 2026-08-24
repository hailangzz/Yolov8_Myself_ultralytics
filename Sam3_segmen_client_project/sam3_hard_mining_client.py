#!/usr/bin/env python3


import argparse
import base64
import json
import os
import re

import requests

############################################
# natural sort
############################################


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


############################################
# SAM3 Client
############################################


class SAM3HardMiningClient:
    def __init__(self, server="http://127.0.0.1:9000", model="segment_anything_3_video"):

        self.server = server.rstrip("/")
        self.model = model

    ########################################
    # image list
    ########################################

    def load_images(self, root):

        data = {"positive": [], "negative": []}

        exist_dir = os.path.join(root, "exist")

        middle_dir = os.path.join(root, "middle")

        null_dir = os.path.join(root, "null")

        def scan(folder):

            result = []

            if not os.path.exists(folder):
                return result

            for name in sorted(os.listdir(folder), key=natural_sort_key):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    result.append(os.path.join(folder, name))

            return result

        exist = scan(exist_dir)

        middle = scan(middle_dir)

        null = scan(null_dir)

        data["positive"] = exist

        data["negative"] = middle + null

        print("======================")

        print("exist:", len(exist))

        print("middle:", len(middle))

        print("null:", len(null))

        print("total:", len(exist) + len(middle) + len(null))

        print("======================")

        return data

    ########################################
    # image encode
    ########################################

    def encode_image(self, path):

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    ########################################
    # SAM3 init
    ########################################

    def init_sequence(self, image):

        payload = {
            "model": self.model,
            "frames": [self.encode_image(image)],
            "start_frame_index": 0,
        }

        r = requests.post(self.server + "/v1/video/init", json=payload, timeout=300)

        result = r.json()

        if not result.get("success"):
            raise RuntimeError(result)

        return result["data"]["session_id"]

    ########################################
    # prompt

    ########################################

    def prompt(self, session, target):

        payload = {
            "session_id": session,
            "model": self.model,
            "frame_index": 0,
            "text_prompt": target,
            "obj_id": 1,
        }

        r = requests.post(self.server + "/v1/video/prompt", json=payload, timeout=300)

        result = r.json()

        if not result.get("success", False):
            return []

        masks = result.get("data", {}).get("masks", [])

        return masks

    ########################################
    # SAM3 single image
    ########################################

    def infer_image(self, image, target):

        session = self.init_sequence(image)

        masks = self.prompt(session, target)

        return masks

    ########################################
    # mining
    ########################################

    def run(self, dataset, target, root):

        false_positive = []

        false_negative = []

        total = 0

        ####################################
        # small model positive
        ####################################

        print("\nProcessing exist...")

        for img in dataset["positive"]:
            total += 1

            masks = self.infer_image(img, target)

            if len(masks) == 0:
                false_positive.append({"image": os.path.relpath(img, root), "sam3_mask_count": 0})

        ####################################
        # small model negative
        ####################################

        print("\nProcessing negative...")

        for img in dataset["negative"]:
            total += 1

            masks = self.infer_image(img, target)

            if len(masks) > 0:
                false_negative.append({"image": os.path.relpath(img, root), "sam3_mask_count": len(masks)})

        return {
            "false_positive": false_positive,
            "false_negative": false_negative,
            "total": total,
        }

    ########################################
    # save json
    ########################################

    def save_json(self, result, target, root, output):

        total = result["total"]

        fp = result["false_positive"]

        fn = result["false_negative"]

        data = {
            "summary": {
                "target": target,
                "image_root": root,
                "total_images": total,
                "false_positive_count": len(fp),
                "false_negative_count": len(fn),
                "false_positive_rate": round(len(fp) / total, 6),
                "false_negative_rate": round(len(fn) / total, 6),
            },
            "false_positive": fp,
            "false_negative": fn,
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("\nResult saved:")

        print(output)


############################################
# main
############################################


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", required=True)

    parser.add_argument("--target", required=True)

    parser.add_argument("--server", default="http://172.16.50.229:9000")

    parser.add_argument("--output", default="sam3_hard_mining.json")

    args = parser.parse_args()

    client = SAM3HardMiningClient(server=args.server)

    dataset = client.load_images(args.image_root)

    result = client.run(dataset, args.target, args.image_root)

    client.save_json(result, args.target, args.image_root, args.output)


if __name__ == "__main__":
    main()

# 示例：python sam3_hard_mining_client.py --image_root /home/chenkejing/Desktop/CarpetSegmentProject/spatial_location_val_images/carpet_detect/A10-25-YD-005002-test/20260730 --target carpet
# 示例：python sam3_hard_mining_client.py --image_root s3://robot-ai-platform/datasets/carpet_detection/source/images/A10-25-YD-005002-test/20260730/ --target carpet --server http://
