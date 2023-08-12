from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
import imagehash

app = FastAPI()

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

model = nn.Sequential(*list(model.children())[:-2])


def preprocess_image(input_image):
    input_image = input_image.convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)

    return input_tensor


def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm_vector1 = torch.norm(vector1)
    norm_vector2 = torch.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


@app.get("/")
def start(response: Response):
    response.status_code = 201
    return {"status": "Backend is Running"}


@app.post("/api/isSimilar")
def isSimilar(response: Response, remaining_data, file: UploadFile):
    try:
        input_image = Image.open(io.BytesIO(file.file.read()))
        input_tensor = preprocess_image(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            features = model(input_batch)

        flattened_features = features.view(features.size(0), -1)
        hash_code = imagehash.phash(input_image)

        if len(remaining_data) > 0:
            for i in remaining_data:
                similarity_score = cosine_similarity(flattened_features[0], i[0][0])
                hamming_distance = imagehash.ImageHash.__sub__(hash_code, i[1])
                if similarity_score > 0.95 or hamming_distance > 5:
                    response.status_code = 401
                    return {"Status": "Similarity Detected"}

            response.status_code = 201
            return {"Status": "No Similarity Detected"}

        response.status_code = 201
        return {"Status": "No data present to compare"}

    except Exception as e:
        response.status_code = 401
        return {"Error": e}
