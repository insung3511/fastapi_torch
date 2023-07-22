from model.deep_learning_model import MNIST_Classify_Model, DataPreprocessing
import numpy as np
import torch
import base64
import cv2

from pydantic import BaseModel
from fastapi import FastAPI

device = torch.device('mps')
SAVED_MODEL_PATH = "./model/model.pth"

CLASSIFY_MODEL = MNIST_Classify_Model().to(device)
CLASSIFY_MODEL.load_state_dict(torch.load(SAVED_MODEL_PATH))

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 28, 28, 1

app = FastAPI()
test_image = cv2.imread("./test.png", cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28, 28))
test_image = base64.b64encode(test_image)
print(test_image)

class RequestInput(BaseModel):
    input: str

@app.get("/")
async def index():
    return {"Message": ["Hello World"]}

@app.post("/predict")
async def predict(request: RequestInput):
    print(request.input)
    request_input = DataPreprocessing(
        target_datatype=np.float32, 
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(request.input)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    return {"prediction": prediction.tolist()}