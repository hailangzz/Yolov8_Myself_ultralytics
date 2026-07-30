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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

########################################
# natural sort
########################################

def natural_sort_key(s):

    return [

        int(text) if text.isdigit() else text.lower()

        for text in re.split(
            "([0-9]+)",
            s
        )

    ]
class SAM3VideoClient:

    def __init__(
            self,
            server="http://127.0.0.1:9000",
            model="segment_anything_3_video",
            enable_mask_filter=False,
            encode_workers=8,
            prompt_workers=8
    ):

        self.server = server.rstrip("/")
        self.model = model

        self.enable_mask_filter = enable_mask_filter

        self.encode_workers = encode_workers

        self.prompt_workers = prompt_workers

        self.mask_filter = MaskFilter(
            top_y_ratio=0.5
        )

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
                os.listdir(image_dir),
                key=natural_sort_key
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

    def encode_images_parallel(
            self,
            images
    ):

        print(
            "Encoding images:",
            len(images)
        )

        with ThreadPoolExecutor(
                max_workers=self.encode_workers
        ) as executor:
            frames = list(
                executor.map(
                    self.encode_image,
                    images
                )
            )

        return frames

    ########################################
    # init
    ########################################

    def init_sequence(
            self,
            images
    ):

        frames = self.encode_images_parallel(
            images
        )

        payload = {

            "model":
                self.model,

            "frames":
                frames,

            "start_frame_index":
                0
        }

        r = requests.post(
            self.server + "/v1/video/init",
            json=payload,
            timeout=300
        )

        result = r.json()

        if not result["success"]:
            raise RuntimeError(result)

        return result["data"]["session_id"]

    def search_prompt_parallel(

            self,

            session_id,

            target,

            frame_num

    ):

        def worker(i):

            result = self.prompt_text(

                session_id,

                target,

                frame_index=i

            )

            if result is None:
                return -1

            masks = result.get(
                "data",
                {}
            ).get(
                "masks",
                []
            )

            if len(masks) > 0:
                return i

            return -1

        with ThreadPoolExecutor(
                max_workers=self.prompt_workers
        ) as executor:

            results = list(

                executor.map(

                    worker,

                    range(frame_num)

                )

            )

        for r in results:

            if r >= 0:
                return r

        return -1

    def process_one_chunk(

            self,

            images,

            start,

            end,

            target,

            class_id

    ):

        print(
            "Start chunk:",
            start,
            end
        )

        chunk_images = images[start:end]

        session = self.init_sequence(
            chunk_images
        )

        prompt_frame = self.search_prompt_parallel(

            session,

            target,

            len(chunk_images)

        )

        if prompt_frame < 0:
            print(
                "No object:",
                start,
                end
            )

            return

        task = self.propagate(

            session,

            len(chunk_images),

            prompt_frame

        )

        result = self.wait(
            task
        )

        ################################
        # 修正全局frame编号
        ################################

        new_result = {}

        for k, v in result["results"].items():
            new_result[
                str(
                    int(k) + start
                )
            ] = v

        result["results"] = new_result

        self.save_results(

            result,

            images,

            class_id

        )

        print(
            "Finished:",
            start,
            end
        )

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

            chunk_size=50,

            class_id=0,

            workers=2

    ):

        jobs = []

        for start in range(

                0,

                len(images),

                chunk_size

        ):
            end = min(

                start + chunk_size,

                len(images)

            )

            jobs.append(

                (
                    start,
                    end
                )

            )

        print(
            "Total chunks:",
            len(jobs)
        )

        with ThreadPoolExecutor(

                max_workers=workers

        ) as executor:

            futures = []

            for start, end in jobs:
                futures.append(

                    executor.submit(

                        self.process_one_chunk,

                        images,

                        start,

                        end,

                        target,

                        class_id

                    )

                )

            for f in as_completed(futures):

                try:

                    f.result()


                except Exception as e:

                    print(
                        "Worker error:",
                        e
                    )



############################################
# main
############################################


if __name__=="__main__":


    ########################################
    # 目标类别
    # SAM3 text prompt输入
    ########################################

    TARGET = "person"



    ########################################
    # 每个chunk包含的帧数量
    #
    # 例如:
    # CHUNK_SIZE=2
    #
    # 表示:
    # 每2张图片创建一个SAM3 video session
    #
    # 图片较少:
    #     可以设置大一些，例如50~100
    #
    # 图片很多(几千张):
    #     建议设置30~100
    #
    ########################################

    CHUNK_SIZE = 2



    ########################################
    # YOLOv8-seg类别编号
    #
    # 例如:
    # person -> 0
    # car    -> 1
    # dog    -> 2
    #
    # 保存label时:
    # 第一列就是该class_id
    ########################################

    CLASS_ID = 0



    ########################################
    # 创建SAM3客户端
    ########################################

    client = SAM3VideoClient(

        # SAM3 Video Server地址
        server=
        "http://172.16.50.229:9000",


        # 是否启用mask后处理过滤
        #
        # True:
        #     使用MaskFilter过滤异常mask
        #
        # False:
        #     直接使用SAM3输出
        #
        enable_mask_filter=False,


        ####################################
        # 图片base64编码线程数
        #
        # 作用:
        #     加速图片读取和编码
        #
        # CPU核心较多:
        #     可以设置8~16
        ####################################

        encode_workers=8,



        ####################################
        # prompt搜索线程数
        #
        # 作用:
        #     同一个chunk内，
        #     并行搜索包含目标的frame
        #
        # 数值越大:
        #     prompt寻找越快
        #
        # 注意:
        #     会增加服务端请求压力
        ####################################

        prompt_workers=8

    )



    ########################################
    # 加载图片
    #
    # 已经经过natural_sort排序
    #
    # 例如:
    #
    # image1.jpg
    # image2.jpg
    # image10.jpg
    #
    # 会按照数字顺序排列
    ########################################

    images = client.load_images(
        "./images"
    )



    ########################################
    # 开始分chunk执行SAM3 Video推理
    #
    # workers:
    #     chunk并行数量
    #
    # 例如:
    #
    # workers=2
    #
    # 同时运行:
    #
    # chunk1:
    #   frame 0-1
    #
    # chunk2:
    #   frame 2-3
    #
    # 两个SAM3 session同时执行
    #
    # 注意:
    #     数值过大会增加GPU显存占用
    ########################################

    client.process_chunks(

        images,

        TARGET,

        # 每多少帧作为一个video chunk
        chunk_size=CHUNK_SIZE,

        # YOLO类别编号
        class_id=CLASS_ID,

        # chunk并行线程数量
        workers=2

    )