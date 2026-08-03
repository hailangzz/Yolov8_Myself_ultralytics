import cv2
import numpy as np


def mask_to_yolo_polygon(
        mask,
        class_id=0
):

    """
    mask:
        H x W numpy array

    return:
        YOLO-seg polygon string
    """


    h, w = mask.shape


    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    polygons = []


    for contour in contours:

        if len(contour) < 3:
            continue

        # 过滤掉mark边缘区域过多的点
        epsilon = (
                0.002 *
                cv2.arcLength(
                    contour,
                    True
                )
        )

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        polygon = []


        for point in contour:

            x, y = point[0]


            polygon.append(
                x / w
            )

            polygon.append(
                y / h
            )


        line = (
            str(class_id)
            + " "
            + " ".join(
                map(
                    lambda x:f"{x:.6f}",
                    polygon
                )
            )
        )


        polygons.append(line)


    return polygons