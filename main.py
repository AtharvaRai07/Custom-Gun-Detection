import io
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
from config.constants import *
from config.paths_config import MODEL_SAVE_PATH

def load_model(path, num_classes, device):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_SAVE_PATH, MODEL_CLASSES, DEVICE)
print(f"Model loaded on: {DEVICE}")

app = FastAPI()

def predict_and_draw(image: Image.Image, threshold=0.5):
    img_rgb    = image.convert("RGB")
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes  = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    draw = ImageDraw.Draw(img_rgb)
    count = 0
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 15), f"Gun {score:.2f}", fill="red")
            count += 1

    print(f"Detections: {count}")
    return img_rgb

@app.get("/")
def read_root():
    return {"message": "Guns Detection API. POST an image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image      = Image.open(io.BytesIO(image_data))
    output     = predict_and_draw(image)

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
