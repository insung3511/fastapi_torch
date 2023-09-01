# FastAPI + Pytorch Deep Learning

I have implemented a Python solution using a pre-trained artificial intelligence model trained with PyTorch. The purpose of this implementation is to serve AI results efficiently, minimizing latency. Initially, I considered using Django RESTful API, but I concluded that it was unnecessary for AI development and API deployment. Therefore, I switched to using FastAPI for this purpose.

## Create Pre-trained model (w/ Pytorch)

I have created a pre-trained model using PyTorch, but I understand that it is not mandatory to use PyTorch. You are free to use your preferred framework. Initially, I built a simple artificial intelligence model to classify the MNIST dataset. The dataset is obtained from the `torchvision` module. Please refer to the `./model` directory in this repository for more details.

To improve convenience and code reusability, the deep learning network is separated into `classify_model.py`, while the code for model training is written in `classify_train.py`. Currently, the model's performance is not given much importance, and it may demonstrate low accuracy. However, this can be improved according to specific requirements. Once the model training is completed, it will return `model.pth`, which is essential for making predictions in the API.

Please make sure to have `model.pth` available, as it will be used for API predictions.

## FastAPI Server

The FastAPI server code is located in `main.py`. The following code snippet was included for temporary testing purposes, and it can be removed if necessary:

```python
test_image = cv2.imread("./test.png", cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28, 28))
test_image = base64.b64encode(test_image)
print(test_image)
```

The reason for using this code was to test whether the server correctly receives Base64-encoded image data as input and returns the expected output. Although it can be removed, it might be useful for future model testing and API verification. Currently, the server retrieves model prediction results by making a request to `{SERVER_HOST}:{SERVER_PORT}/predict`. If you need to handle multiple input values, you can modify the code as follows:

```python
class RequestInput(BaseModel):
    image: str  # Base64 Encoded
    image_name: str
    image_captured_location: str
    ...
```

Please note that the code for applying the model will vary depending on the API requirements, so make sure to customize it accordingly.
