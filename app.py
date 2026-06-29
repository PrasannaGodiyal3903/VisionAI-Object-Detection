from PIL import Image
import io
import pandas as pd
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from pathlib import Path

# -------------------- MODEL --------------------

MODEL_PATH = Path("./models/sample_model/yolov8n.pt")

if MODEL_PATH.exists():
    model_sample_model = YOLO(str(MODEL_PATH))
else:
    print("Downloading yolov8n.pt...")
    model_sample_model = YOLO("yolov8n.pt")


# -------------------- IMAGE HELPERS --------------------

def get_image_from_bytes(binary_image: bytes) -> Image:
    return Image.open(io.BytesIO(binary_image)).convert("RGB")


def get_bytes_from_image(image: Image) -> bytes:
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG", quality=85)
    img_bytes.seek(0)
    return img_bytes


# -------------------- PREDICTION --------------------

def transform_predict_to_df(results, labels_dict):

    predict_bbox = pd.DataFrame(
        results[0].boxes.xyxy.cpu().numpy(),
        columns=["xmin", "ymin", "xmax", "ymax"],
    )

    predict_bbox["confidence"] = (
        results[0].boxes.conf.cpu().numpy()
    )

    predict_bbox["class"] = (
        results[0].boxes.cls.cpu().numpy().astype(int)
    )

    predict_bbox["name"] = predict_bbox["class"].replace(
        labels_dict
    )

    return predict_bbox


def get_model_predict(
    model: YOLO,
    input_image: Image,
    image_size: int = 640,
    conf: float = 0.5,
    save: bool = False,
):

    results = model.predict(
        source=input_image,
        imgsz=image_size,
        conf=conf,
        save=save,
        verbose=False,
    )

    return transform_predict_to_df(
        results,
        model.names,
    )


# -------------------- DRAW BOXES --------------------

def add_bboxs_on_img(
    image: Image,
    predict: pd.DataFrame,
):

    annotator = Annotator(np.array(image))

    predict = predict.sort_values(
        by=["xmin"],
        ascending=True,
    )

    for _, row in predict.iterrows():

        bbox = [
            row["xmin"],
            row["ymin"],
            row["xmax"],
            row["ymax"],
        ]

        label = (
            f"{row['name']} "
            f"{row['confidence']*100:.1f}%"
        )

        annotator.box_label(
            bbox,
            label,
            color=colors(
                int(row["class"]),
                True,
            ),
        )

    return Image.fromarray(
        annotator.result()
    )


# -------------------- MODEL WRAPPER --------------------

def detect_sample_model(
    input_image: Image,
    confidence: float = 0.5,
):

    return get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        image_size=640,
        conf=confidence,
        save=False,
    )