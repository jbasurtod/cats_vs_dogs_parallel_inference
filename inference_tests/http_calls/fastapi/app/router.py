# --- 📦 Imports ---
from fastapi import APIRouter, HTTPException
from models.request import RequestModel
from models.response import ResponseModel
import torch
from torchvision import models, transforms
from PIL import Image
import time
import os
import uuid  # <-- importante

# --- Generate a unique ID per instance ---
INSTANCE_ID = str(uuid.uuid4())

# --- Initialize Router ---
router = APIRouter()

# --- Global model loading ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

model = models.mobilenet_v3_large(pretrained=False)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("/app/model/mobilenetv3_cats_vs_dogs.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Helper function: inference ---
def infer(image_path: str):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()

# --- API Endpoint: /predict ---
@router.post("/predict", response_model=ResponseModel)
def predict(request: RequestModel):
    start_time = time.time()

    image_path = os.path.join('/app/', request.image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        pred_label = infer(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    elapsed_time = time.time() - start_time

    response = ResponseModel(
        image_name=request.image_name,
        time_elapsed=elapsed_time,
        pred_label=pred_label,
        api_id=INSTANCE_ID  # 🔥 Aquí usamos el UUID generado
    )

    return response
