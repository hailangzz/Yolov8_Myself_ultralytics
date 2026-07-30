#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import base64
import requests
import time

import numpy as np
from PIL import Image

import cv2
from utils.mask_filter import MaskFilter


class SAM3VideoClient:


    def __init__(
        self,
        server="http://127.0.0.1:9000",
        model="segment_anything_3_video",
        enable_mask_filter=False
    ):
        self.server=server.rstrip("/")
        self.model=model

        # mask过滤器
        self.enable_mask_filter = enable_mask_filter
        self.mask_filter = MaskFilter(top_y_ratio=0.5)

    ########################################
    # image -> base64
    ########################################

    def encode_image(
        self,
        path
    ):

        with open(path,"rb") as f:

            return base64.b64encode(
                f.read()
            ).decode()



    ########################################
    # load images
    ########################################

    def load_images(
        self,
        image_dir
    ):


        images=[]


        for name in sorted(
            os.listdir(image_dir)
        ):

            if name.lower().endswith(
                (
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            ):

                images.append(
                    os.path.join(
                        image_dir,
                        name
                    )
                )


        print(
            "frames:",
            len(images)
        )


        return images




    ########################################
    # init
    ########################################

    def init_sequence(
        self,
        images
    ):


        frames=[]


        for img in images:

            frames.append(
                self.encode_image(img)
            )


        payload={

            "model":
                self.model,

            "frames":
                frames,

            "start_frame_index":
                0

        }


        r=requests.post(

            self.server+
            "/v1/video/init",

            json=payload,

            timeout=300

        )


        result=r.json()


        print(
            "INIT:",
            result
        )


        if not result.get("success"):

            raise RuntimeError(
                result
            )


        return result["data"]["session_id"]





    ########################################
    # text prompt
    ########################################

    def prompt_text(

        self,

        session_id,

        text,

        frame_index=0

    ):


        payload={


            "session_id":
                session_id,


            "model":
                self.model,


            "frame_index":
                frame_index,


            "text_prompt":
                text,


            "obj_id":
                1

        }


        r=requests.post(

            self.server+
            "/v1/video/prompt",

            json=payload,

            timeout=300

        )


        result=r.json()


        print(
            "PROMPT:",
            result
        )


        if not result.get(
            "success",
            False
        ):

            return None



        masks=result.get(
            "data",
            {}
        ).get(
            "masks",
            []
        )

        if len(masks) == 0:
            return None


        return result






    ########################################
    # propagate
    ########################################

    ########################################
    # propagate
    ########################################

    def propagate(

            self,

            session_id,

            num_frames,

            start_frame=0

    ):

        payload = {

            "session_id":
                session_id,

            "model":
                self.model,

            "start_frame":
                start_frame,

            "end_frame":
                num_frames - 1

        }

        r = requests.post(

            self.server +
            "/v1/video/propagate",

            json=payload,

            timeout=300

        )

        result = r.json()

        print(
            "PROPAGATE:",
            result
        )

        if not result.get(
                "success",
                False
        ):
            raise RuntimeError(

                result.get(
                    "error",
                    {}
                ).get(
                    "message",
                    "propagate failed"
                )

            )

        return result["data"]["task_id"]






    ########################################
    # wait
    ########################################

    def wait(

        self,

        task_id

    ):


        while True:


            r=requests.get(

                self.server+
                "/v1/video/status/"
                +
                task_id,

                timeout=60

            )


            result=r.json()


            print(
                result
            )


            data=result.get(
                "data",
                {}
            )


            status=data.get(
                "status"
            )


            if status=="completed":

                return data



            if status in [
                "failed",
                "cancelled"
            ]:

                raise RuntimeError(
                    result
                )



            time.sleep(2)





    ########################################
    # save mask + YOLOv8-seg
    ########################################

    ########################################
    # save mask + YOLOv8-seg
    # detected / undetected
    ########################################

    def save_results(

            self,

            result,

            image_paths,

            class_id=0

    ):

        ####################################
        # 创建父目录
        ####################################

        detected_root = "./result/detected"

        undetected_root = "./result/undetected"

        detected_vis = os.path.join(
            detected_root,
            "vis"
        )

        detected_label = os.path.join(
            detected_root,
            "labels"
        )

        undetected_vis = os.path.join(
            undetected_root,
            "vis"
        )

        undetected_label = os.path.join(
            undetected_root,
            "labels"
        )

        for d in [

            detected_vis,
            detected_label,
            undetected_vis,
            undetected_label

        ]:
            os.makedirs(
                d,
                exist_ok=True
            )

        ####################################
        # 获取结果
        ####################################

        results = result.get(
            "results",
            {}
        )

        print(
            "RESULT FRAMES:",
            len(results)
        )

        ####################################
        # 遍历每帧
        ####################################

        for frame_id, frame_data in results.items():

            try:

                idx = int(frame_id)

                img = Image.open(
                    image_paths[idx]
                ).convert(
                    "RGB"
                )

                img = np.array(img)

                h, w = img.shape[:2]

                overlay = img.copy()

                masks = frame_data.get(
                    "masks",
                    []
                )

                ################################
                # SAM3 mask过滤
                ################################

                if self.enable_mask_filter:

                    print(
                        "[MaskFilter] enabled"
                    )

                    masks = self.mask_filter.filter_masks(
                        masks,
                        w,
                        h
                    )

                else:

                    print(
                        "[MaskFilter] disabled"
                    )

                ################################
                # 是否检测到目标
                ################################

                detected = (

                        masks is not None

                        and

                        len(masks) > 0

                )

                yolo_lines = []

                ################################
                # 处理mask
                ################################

                for obj in masks:

                    points = obj.get(
                        "points",
                        []
                    )

                    label = obj.get(
                        "label",
                        "obj"
                    )

                    score = obj.get(
                        "score",
                        0
                    )

                    if len(points) < 3:
                        continue

                    polygon = np.array(

                        points,

                        dtype=np.int32

                    )

                    ################################
                    # YOLOv8-seg格式
                    ################################

                    yolo_points = []

                    for x, y in polygon:
                        nx = float(x) / w

                        ny = float(y) / h

                        yolo_points.append(

                            f"{nx:.6f}"

                        )

                        yolo_points.append(

                            f"{ny:.6f}"

                        )

                    yolo_lines.append(

                        str(class_id)

                        +

                        " "

                        +

                        " ".join(
                            yolo_points
                        )

                    )

                    ################################
                    # 绘制mask
                    ################################

                    mask = np.zeros(

                        img.shape[:2],

                        dtype=np.uint8

                    )

                    cv2.fillPoly(

                        mask,

                        [
                            polygon
                        ],

                        255

                    )

                    color = np.zeros_like(
                        img
                    )

                    color[:, :, 1] = 255

                    alpha = 0.4

                    area = mask > 0

                    overlay[area] = (

                            img[area] * (1 - alpha)

                            +

                            color[area] * alpha

                    )

                    cv2.polylines(

                        overlay,

                        [
                            polygon
                        ],

                        True,

                        (255, 0, 0),

                        3

                    )

                    x0, y0 = polygon[0]

                    cv2.putText(

                        overlay,

                        f"{label}:{score:.2f}",

                        (
                            int(x0),
                            int(y0)
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1,

                        (255, 255, 255),

                        2

                    )

                ####################################
                # 选择保存目录
                ####################################

                if detected:

                    vis_dir = detected_vis

                    label_dir = detected_label


                else:

                    vis_dir = undetected_vis

                    label_dir = undetected_label

                ####################################
                # 保存可视化图片
                # 原图名称 + _mask
                ####################################

                src_name = os.path.basename(

                    image_paths[idx]

                )

                name, ext = os.path.splitext(

                    src_name

                )

                img_name = name + "_mask" + ext

                img_save = os.path.join(

                    vis_dir,

                    img_name

                )

                Image.fromarray(
                    overlay
                ).save(
                    img_save
                )

                ####################################
                # 保存YOLO label
                ####################################

                txt_name = os.path.splitext(

                    os.path.basename(
                        image_paths[idx]
                    )

                )[0] + ".txt"

                txt_save = os.path.join(

                    label_dir,

                    txt_name

                )

                with open(

                        txt_save,

                        "w"

                ) as f:

                    for line in yolo_lines:
                        f.write(

                            line + "\n"

                        )

                if detected:

                    print(

                        "[DETECTED]",

                        img_save,

                        txt_save

                    )


                else:

                    print(

                        "[UNDETECTED]",

                        img_save,

                        txt_save

                    )



            except Exception as e:

                print(

                    "frame error:",

                    frame_id,

                    e

                )

    ########################################
    # chunk video processing
    ########################################

    def process_chunks(

            self,

            images,

            target,

            chunk_size=100,

            class_id=0

    ):

        total_frames = len(images)

        print(
            "Total frames:",
            total_frames
        )

        ####################################
        # 分chunk
        ####################################

        for start in range(
                0,
                total_frames,
                chunk_size
        ):

            end = min(

                start + chunk_size,

                total_frames

            )

            print(
                "\n===================="
            )

            print(
                "Processing:",
                start,
                "~",
                end - 1
            )

            ################################
            # 当前chunk图片
            ################################

            chunk_images = images[start:end]

            ################################
            # init
            ################################

            session = self.init_sequence(

                chunk_images

            )

            ################################
            # 搜索prompt帧
            ################################

            prompt = None

            prompt_frame = -1

            for i in range(

                    len(chunk_images)

            ):

                print(

                    "Prompt search:",
                    start + i

                )

                prompt = self.prompt_text(

                    session,

                    target,

                    frame_index=i

                )

                if prompt is None:
                    continue

                masks = prompt.get(

                    "data",

                    {}

                ).get(

                    "masks",

                    []

                )

                if len(masks) > 0:
                    prompt_frame = i

                    print(

                        "Found object:",
                        start + i

                    )

                    break

            ################################
            # 当前chunk无目标
            ################################

            if prompt_frame < 0:
                print(

                    "No object in chunk:",
                    start,
                    end

                )

                continue

            ################################
            # propagate
            ################################

            task = self.propagate(

                session,

                len(chunk_images),

                start_frame=prompt_frame

            )

            result = self.wait(

                task

            )

            ################################
            # 调整frame索引
            ################################

            new_result = {}

            for k, v in result.get(

                    "results",

                    {}

            ).items():
                global_index = int(k) + start

                new_result[

                    str(global_index)

                ] = v

            result["results"] = new_result

            ################################
            # 保存
            ################################

            self.save_results(

                result,

                images,

                class_id

            )

            print(

                "Finished chunk:",

                start,

                end

            )




############################################
# main
############################################


if __name__=="__main__":

    ################################
    # 配置
    ################################

    TARGET = "person"

    CHUNK_SIZE = 100

    CLASS_ID = 0

    enable_mask_filter_switch = False


    client=SAM3VideoClient(

        server=
        "http://172.16.50.229:9000",
        enable_mask_filter=enable_mask_filter_switch

    )


    image_dir="./images"


    images=client.load_images(
        image_dir
    )

    ################################
    # 分chunk执行SAM3
    ################################

    client.process_chunks(

        images,

        TARGET,

        chunk_size=CHUNK_SIZE,

        class_id=CLASS_ID

    )