#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import base64
import requests
import time


class SAM3VideoClient:


    def __init__(
        self,
        server="http://127.0.0.1:9000",
        model="segment_anything_3_video"
    ):

        self.server = server.rstrip("/")
        self.model = model



    def encode_image(self, image_path):

        with open(image_path, "rb") as f:
            img = f.read()

        return base64.b64encode(img).decode()



    def load_images(self, image_dir):

        images=[]


        for name in sorted(
            os.listdir(image_dir)
        ):

            if name.lower().endswith(
                (".jpg",".jpeg",".png")
            ):

                images.append(
                    os.path.join(
                        image_dir,
                        name
                    )
                )


        print(
            f"Found {len(images)} images"
        )

        return images



    ########################################
    # 初始化多图片序列
    ########################################

    def init_sequence(
        self,
        image_paths
    ):

        print(
            "Encoding images..."
        )


        frames=[]


        for img in image_paths:

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


        r.raise_for_status()


        result=r.json()


        print(result)


        session_id = (
            result["data"]["session_id"]
        )


        print(
            "session:",
            session_id
        )


        return session_id



    ########################################
    # 添加点提示
    ########################################

    def prompt_point(

        self,
        session_id,

        x,

        y,

        frame_index=0

    ):


        payload={


            "session_id":
                session_id,


            "model":
                self.model,


            "frame_index":
                frame_index,


            "points":[
                [
                    x,
                    y
                ]
            ],


            "point_labels":[
                1
            ],


            "obj_id":
                1

        }


        r=requests.post(

            self.server+
            "/v1/video/prompt",

            json=payload,

            timeout=300

        )


        print(
            r.json()
        )


        return r.json()



    ########################################
    # 开始传播
    ########################################

    def propagate(

        self,

        session_id,

        num_frames

    ):


        payload={


            "session_id":
                session_id,


            "model":
                self.model,


            "start_frame":
                0,


            "end_frame":
                num_frames-1

        }


        r=requests.post(

            self.server+
            "/v1/video/propagate",

            json=payload,

            timeout=300

        )


        result=r.json()


        print(result)


        return (
            result["data"]["task_id"]
        )



    ########################################
    # 等待完成
    ########################################

    def wait(

        self,

        task_id

    ):


        while True:


            r=requests.get(

                self.server+
                f"/v1/video/status/{task_id}"

            )


            result=r.json()


            print(result)


            status=result.get(
                "data",
                {}
            ).get(
                "status"
            )


            if status in [
                "completed",
                "failed"
            ]:

                return result


            time.sleep(2)



#############################################
# 测试
#############################################


if __name__=="__main__":


    client=SAM3VideoClient(

        server=
        "http://172.16.50.229:9000"

    )


    #
    # 图片目录
    #

    image_dir="./images"


    images=client.load_images(
        image_dir
    )


    #
    # 初始化
    #

    session_id = client.init_sequence(
        images
    )


    #
    # 第一帧目标点
    #

    client.prompt_point(

        session_id,

        x=300,

        y=200

    )


    #
    # 开始传播
    #

    task_id=client.propagate(

        session_id,

        len(images)

    )


    #
    # 等待
    #

    result=client.wait(
        task_id
    )


    print(
        "DONE"
    )

    print(result)