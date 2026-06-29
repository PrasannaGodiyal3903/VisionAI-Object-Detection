####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys

from fastapi import FastAPI, File, Query, status
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from app import (
    get_image_from_bytes,
    detect_sample_model,
    add_bboxs_on_img,
    get_bytes_from_image,
)

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

app = FastAPI(
    title="VisionAI Detection API",
    description="Real-time object detection API powered by YOLOv8 and FastAPI.",
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8008",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def save_openapi_json():
    openapi_data = app.openapi()
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def perform_healthcheck():
    return {"healthcheck": "Everything OK!"}


######################### Support Func #################################


def crop_image_by_predict(
    image: Image,
    predict: pd.DataFrame(),
    crop_class_name: str,
) -> Image:

    crop_predicts = predict[(predict["name"] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(
            status_code=400,
            detail=f"{crop_class_name} not found in photo",
        )

    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(
            by=["confidence"],
            ascending=False,
        )

    crop_bbox = crop_predicts[
        ["xmin", "ymin", "xmax", "ymax"]
    ].iloc[0].values

    img_crop = image.crop(crop_bbox)

    return img_crop


######################### API #################################


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(
    file: bytes = File(...),
    confidence: float = Query(0.5),
):
    """
    Detect objects and return JSON.
    """

    result = {"detect_objects": None}

    input_image = get_image_from_bytes(file)

    predict = detect_sample_model(
        input_image,
        confidence=confidence,
    )

    detect_res = predict[["name", "confidence"]]

    objects = detect_res["name"].values

    result["detect_objects_names"] = ", ".join(objects)

    result["detect_objects"] = json.loads(
        detect_res.to_json(orient="records")
    )

    logger.info("results: {}", result)

    return result


@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(
    file: bytes = File(...),
    confidence: float = Query(0.5),
):
    """
    Detect objects and return annotated image.
    """

    input_image = get_image_from_bytes(file)

    predict = detect_sample_model(
        input_image,
        confidence=confidence,
    )

    final_image = add_bboxs_on_img(
        image=input_image,
        predict=predict,
    )

    return StreamingResponse(
        content=get_bytes_from_image(final_image),
        media_type="image/jpeg",
    )