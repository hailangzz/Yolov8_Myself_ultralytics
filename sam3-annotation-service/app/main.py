from fastapi import FastAPI
from pydantic import BaseModel

from app.sam3_service import SAM3Service
from app.converters.mask_to_yolo import mask_to_yolo_polygon
from app.config_loader import load_config
from app.utils.file import save_yolo_label
import os

app = FastAPI(
    title="SAM3 Annotation Service"
)


config = load_config()
sam3 = SAM3Service(
    model_path=config["model"]["path"],
    conf=config["sam3"]["conf"],
    quantize=config["sam3"]["quantize"],
    device=config["runtime"]["device"]
)



class InferenceRequest(BaseModel):

    image_path:str

    prompts:list[str]



@app.post("/v1/inference")
def inference(
    req:InferenceRequest
):


    results = sam3.predict(
        req.image_path,
        req.prompts
    )


    result = results[0]


    objects = 0

    yolo_labels = []


    if result.masks is not None:


        masks = (
            result.masks.data
            .cpu()
            .numpy()
        )


        objects = len(masks)


        for mask in masks:


            labels = mask_to_yolo_polygon(
                mask,
                class_id=0
            )


            yolo_labels.extend(
                labels
            )




    image_name = os.path.basename(
        req.image_path
    )

    image_id = os.path.splitext(
        image_name
    )[0]


    label_path = (
        f"/app/output/{image_id}.txt"
    )


    save_yolo_label(
        label_path,
        yolo_labels
    )


    return {

        "status":"success",

        "objects":objects,

        "label_path":label_path

    }