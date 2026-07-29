#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import base64
import requests
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2



class SAM3VideoClient:


    def __init__(
        self,
        server="http://127.0.0.1:9000",
        model="segment_anything_3_video"
    ):

        self.server=server.rstrip("/")
        self.model=model



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



        try:


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



            if len(masks)==0:

                print(
                    "No object:",
                    text
                )

                return None



            return result



        except Exception as e:


            print(
                "prompt error:",
                e
            )


            return None






    ########################################
    # propagate
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
    # draw mask polygon
    ########################################


    def save_results(

        self,

        result,

        image_paths,

        output_dir="./vis"

    ):


        os.makedirs(
            output_dir,
            exist_ok=True
        )


        results=result.get(
            "results",
            {}
        )


        print(
            "RESULT FRAMES:",
            len(results)
        )



        for frame_id,frame_data in results.items():


            try:


                idx=int(frame_id)


                img=Image.open(
                    image_paths[idx]
                ).convert(
                    "RGB"
                )


                img=np.array(img)



                overlay=img.copy()



                masks=frame_data.get(
                    "masks",
                    []
                )



                for obj in masks:



                    points=obj.get(
                        "points",
                        []
                    )


                    label=obj.get(
                        "label",
                        "obj"
                    )


                    score=obj.get(
                        "score",
                        0
                    )



                    if len(points)<3:

                        continue



                    polygon=np.array(

                        points,

                        dtype=np.int32

                    )



                    mask=np.zeros(

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



                    color=np.zeros_like(
                        img
                    )


                    color[:,:,1]=255



                    alpha=0.4



                    area=mask>0



                    overlay[area]=(

                        img[area]*(1-alpha)

                        +

                        color[area]*alpha

                    )



                    cv2.polylines(

                        overlay,

                        [
                            polygon
                        ],

                        True,

                        (255,0,0),

                        3

                    )



                    x,y=polygon[0]



                    cv2.putText(

                        overlay,

                        f"{label}:{score:.2f}",

                        (
                            int(x),
                            int(y)
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1,

                        (255,255,255),

                        2

                    )



                save=os.path.join(

                    output_dir,

                    f"{idx:06d}_mask.jpg"

                )


                Image.fromarray(
                    overlay
                ).save(
                    save
                )


                print(
                    "saved:",
                    save
                )



            except Exception as e:


                print(
                    "frame error:",
                    frame_id,
                    e
                )






############################################
# main
############################################


if __name__=="__main__":


    client=SAM3VideoClient(

        server=
        "http://172.16.50.229:9000"

    )


    image_dir="./images"


    images=client.load_images(
        image_dir
    )



    session=client.init_sequence(
        images
    )



    # 修改这里
    target="chair"



    prompt=client.prompt_text(

        session,

        target

    )



    if prompt is None:


        print(
            "目标不存在，退出"
        )

        exit(0)




    task=client.propagate(

        session,

        len(images)

    )



    result=client.wait(
        task
    )



    client.save_results(

        result,

        images,

        "./vis"

    )