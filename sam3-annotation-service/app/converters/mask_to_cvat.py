import cv2


def mask_to_polygon(mask):

    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []

    for c in contours:
        points = []

        for p in c:
            x, y = p[0]

            points.append({"x": int(x), "y": int(y)})

        polygons.append(points)

    return polygons
