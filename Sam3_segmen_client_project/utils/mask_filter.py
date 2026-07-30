#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np



class MaskFilter:


    def __init__(

            self,

            top_y_ratio=0.5

    ):


        """
        top_y_ratio:

            图像高度比例

            0.5表示图像高度中线


            如果mask最高点(y最小值)

            小于:

                image_height*0.5

            则过滤

        """


        self.top_y_ratio = top_y_ratio






    ####################################
    # 过滤单个mask
    ####################################

    def floor_target_y_center_pix_filter(

            self,

            points,

            image_width,

            image_height

    ):


        """
        过滤规则:

        找mask轮廓中 y 最小的点

        如果:

            y_min < image_height*0.5

        过滤


        返回:

            True:
                保留


            False:
                丢弃

        """



        if points is None:

            return False



        if len(points)<3:

            return False



        polygon=np.array(

            points,

            dtype=np.int32

        )



        ################################
        # 找轮廓最高点
        ################################

        top_index=np.argmin(

            polygon[:,1]

        )


        top_point=polygon[

            top_index

        ]



        top_y=top_point[1]




        ################################
        # 高度阈值
        ################################

        limit=image_height*self.top_y_ratio




        ################################
        # 上半区域过滤
        ################################

        if top_y < limit:


            print(

                "[MaskFilter] remove",

                "top_y:",

                top_y,

                "limit:",

                limit

            )


            return False



        return True






    ####################################
    # 批量过滤mask
    ####################################

    def filter_masks(

            self,

            masks,

            image_width,

            image_height

    ):


        results=[]


        for mask in masks:


            points=mask.get(

                "points",

                []

            )


            if self.floor_target_y_center_pix_filter(

                    points,

                    image_width,

                    image_height

            ):


                results.append(

                    mask

                )


        return results