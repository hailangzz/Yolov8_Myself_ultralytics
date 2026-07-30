#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
SAM3 Hard Mining Client

功能:
1. 输入一个包含 exist / middle / null 子目录的数据目录
2. 将:
   exist -> 小模型认为有目标
   middle + null -> 小模型认为无目标/不确定
3. 调用 SAM3 Video 服务推理
4. 挖掘:
   - exist 中 SAM3 未发现目标: 漏检 hard negative
   - middle/null 中 SAM3 发现目标: 漏检 hard positive
5. 输出 json 结果用于小模型优化
"""

import os
import json
import base64
import re
import time
import argparse
import requests


def natural_sort_key(s):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"([0-9]+)", s)
    ]


class SAM3HardMiningClient:

    def __init__(
        self,
        server="http://172.16.50.229:9000",
        model="segment_anything_3_video"
    ):
        self.server = server.rstrip("/")
        self.model = model

    def encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def collect_images(self, image_dir):
        groups = {
            "exist": [],
            "middle": [],
            "null": []
        }

        for cls in groups:
            folder = os.path.join(image_dir, cls)

            if not os.path.exists(folder):
                continue

            for name in sorted(os.listdir(folder), key=natural_sort_key):
                if name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    groups[cls].append(
                        os.path.join(folder, name)
                    )

        return groups

    def init_sequence(self, images):

        payload = {
            "model": self.model,
            "frames": [
                self.encode_image(x)
                for x in images
            ],
            "start_frame_index": 0
        }

        r = requests.post(
            self.server + "/v1/video/init",
            json=payload,
            timeout=300
        )

        result = r.json()

        if not result.get("success"):
            raise RuntimeError(result)

        return result["data"]["session_id"]

    def prompt(self, session_id, text, frame_index):

        payload = {
            "session_id": session_id,
            "model": self.model,
            "frame_index": frame_index,
            "text_prompt": text,
            "obj_id": 1
        }

        r = requests.post(
            self.server + "/v1/video/prompt",
            json=payload,
            timeout=300
        )

        result = r.json()

        if not result.get("success"):
            return []

        return result.get(
            "data",
            {}
        ).get(
            "masks",
            []
        )

    def process_category(
        self,
        images,
        category,
        target
    ):

        results = []

        if len(images) == 0:
            return results

        print(
            "Processing",
            category,
            len(images)
        )

        session = self.init_sequence(images)

        for idx, img in enumerate(images):

            masks = self.prompt(
                session,
                target,
                idx
            )

            sam3_detect = len(masks) > 0

            if category == "exist" and not sam3_detect:
                status = "exist_but_sam3_miss"

            elif category in ["middle", "null"] and sam3_detect:
                status = "sam3_found_but_small_model_miss"

            else:
                continue

            results.append(
                {
                    "image": img,
                    "small_model_category": category,
                    "sam3_detect": sam3_detect,
                    "sam3_mask_count": len(masks),
                    "status": status
                }
            )

        return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        required=True,
        help="包含 exist/middle/null 的目录"
    )

    parser.add_argument(
        "--target",
        default="carpet"
    )

    parser.add_argument(
        "--server",
        default="http://172.16.50.229:9000"
    )

    parser.add_argument(
        "--output",
        default="sam3_hard_mining.json"
    )

    args = parser.parse_args()

    client = SAM3HardMiningClient(
        server=args.server
    )

    groups = client.collect_images(
        args.image_dir
    )

    all_results = []

    # exist: 关注SAM3漏检
    all_results.extend(
        client.process_category(
            groups["exist"],
            "exist",
            args.target
        )
    )

    # middle + null: 关注SAM3发现的新目标
    all_results.extend(
        client.process_category(
            groups["middle"],
            "middle",
            args.target
        )
    )

    all_results.extend(
        client.process_category(
            groups["null"],
            "null",
            args.target
        )
    )

    output = {
        "target": args.target,
        "server": args.server,
        "hard_samples": all_results,
        "count": len(all_results)
    }

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "Saved:",
        args.output
    )


if __name__ == "__main__":
    main()
