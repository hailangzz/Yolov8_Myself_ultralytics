#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 Video Client for X-AnyLabeling Server

API:
    POST /v1/video/init
    POST /v1/video/prompt
    POST /v1/video/propagate
    GET  /v1/video/status/{task_id}

Author:
    X-AnyLabeling SAM3-video Client
"""


import os
import cv2
import time
import base64
import requests

from tqdm import tqdm
import numpy as np
from PIL import Image


class SAM3VideoClient:


    def __init__(
        self,
        server="http://127.0.0.1:9000",
        model="segment_anything_3_video"
    ):

        self.server = server.rstrip("/")
        self.model = model


    #################################################
    # image -> base64
    #################################################

    def image_to_base64(self, image_path):

        with open(image_path, "rb") as f:
            data = f.read()

        return base64.b64encode(data).decode()



    #################################################
    # video -> frames
    #################################################

    def extract_frames(
        self,
        video_path,
        output_dir,
        max_frames=None
    ):

        os.makedirs(
            output_dir,
            exist_ok=True
        )


        cap=cv2.VideoCapture(video_path)


        frame_paths=[]

        index=0


        while True:

            ret,frame=cap.read()

            if not ret:
                break


            filename=os.path.join(
                output_dir,
                f"{index:06d}.jpg"
            )


            cv2.imwrite(
                filename,
                frame
            )


            frame_paths.append(filename)


            index+=1


            if max_frames and index>=max_frames:
                break


        cap.release()


        print(
            f"[INFO] Extract frames: {len(frame_paths)}"
        )


        return frame_paths



    #################################################
    # init video session
    #################################################

    def init_video(
        self,
        frame_paths
    ):


        print(
            "[INFO] Initializing video session..."
        )


        frames=[]


        for p in tqdm(frame_paths):

            frames.append(
                self.image_to_base64(p)
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


        session_id=result["data"]["session_id"]


        print(
            "[INFO] session_id:",
            session_id
        )


        return session_id



    #################################################
    # prompt point
    #################################################

    def add_point_prompt(
        self,
        session_id,
        x,
        y,
        frame_index=0,
        obj_id=1
    ):


        print(
            "[INFO] Add point prompt:",
            x,
            y
        )


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
                obj_id

        }



        r=requests.post(

            self.server+
            "/v1/video/prompt",

            json=payload,

            timeout=300

        )


        r.raise_for_status()


        result=r.json()


        print(result)


        return result



    #################################################
    # propagate
    #################################################

    def propagate(
        self,
        session_id,
        start_frame=0,
        end_frame=None
    ):


        print(
            "[INFO] Start propagation..."
        )


        payload={


            "session_id":
                session_id,


            "model":
                self.model,


            "start_frame":
                start_frame,


            "end_frame":
                end_frame

        }


        r=requests.post(

            self.server+
            "/v1/video/propagate",

            json=payload,

            timeout=300

        )


        r.raise_for_status()


        result=r.json()


        print(result)


        task_id=result["data"]["task_id"]


        print(
            "[INFO] task_id:",
            task_id
        )


        return task_id



    #################################################
    # query status
    #################################################

    def wait_result(
        self,
        task_id,
        interval=3
    ):


        print(
            "[INFO] Waiting..."
        )


        while True:


            r=requests.get(

                self.server+
                f"/v1/video/status/{task_id}",

                timeout=60

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
                "failed",
                "cancelled"
            ]:

                return result



            time.sleep(interval)



    #################################################
    # cleanup
    #################################################

    def cleanup(
        self,
        session_id
    ):


        url=(
            self.server+
            f"/v1/video/cleanup/{session_id}"
        )


        r=requests.post(

            url,

            params={
                "model":self.model
            }

        )


        print(
            r.json()
        )





########################################################
# main test
########################################################


if __name__=="__main__":


    SERVER="http://172.16.50.229:9000"


    client=SAM3VideoClient(
        SERVER
    )


    #
    # 1. 视频拆帧
    #

    frames=client.extract_frames(

        "test.mp4",

        "./frames",

        max_frames=100

    )


    #
    # 2. 初始化
    #

    session_id=client.init_video(
        frames
    )


    #
    # 3. 添加目标点
    #
    # 修改这里
    #

    client.add_point_prompt(

        session_id,

        x=300,

        y=200

    )


    #
    # 4. 开始传播
    #

    task_id=client.propagate(

        session_id,

        0,

        len(frames)-1

    )


    #
    # 5. 等待结果
    #

    result=client.wait_result(
        task_id
    )


    print(
        "FINAL RESULT:"
    )

    print(result)



    #
    # 6. 清理session
    #

    client.cleanup(
        session_id
    )