import io

from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

IMG_SIZE = (64, 64)
model = tf.keras.models.load_model("models/model_loss_0.882523775100708_acc_0.6944937705993652.h5")

app = FastAPI()

RESULTS = ["black", "blue", "green", "yellow"]


def imageThroughIA(image: Image) -> str:
    imag = np.array(image)
    predictions = model.predict(imag[None, :, :]).argmax(axis=1)
    return RESULTS[predictions[0]]


@app.get("/")
async def root():
    return {"message": "Hello There"}


@app.post("/check-recyclable")
async def checkRecyclable(file: UploadFile):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).resize(IMG_SIZE)
    res = imageThroughIA(img)
    return res
