# --- 📦 Imports ---
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import time
import os

# --- 🧠 PyTorch device configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# --- 📦 Load model function ---
def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load("model/mobilenetv3_cats_vs_dogs.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# --- Load model once ---
model = load_model()

# --- 🚀 Single image inference function ---
def infer_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        start = time.time()
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        elapsed = time.time() - start
        pred_label = preds.item()

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        elapsed = 0
        pred_label = -1  # error code

    return elapsed, pred_label

# --- 🚀 Main execution ---
if __name__ == "__main__":
    # --- Load CSV ---
    csv_path = "inference_tests/sequential/csv/training_images.csv"
    df = pd.read_csv(csv_path)

    # --- Filter test images ---
    test_df = df[df['type'] == 'test'].reset_index(drop=True)
    print(f"✅ Total test images found: {len(test_df)}")

    # --- Map classes to numbers ---
    class_to_idx = {"cat": 0, "dog": 1}
    test_df['true_label'] = test_df['class'].map(class_to_idx)

    # --- Prepare full paths ---
    test_df['full_path'] = test_df['image']#.apply(lambda x: os.path.join('/', x) if not x.startswith('/') else x)

    results = []

    # --- Start total timing ---
    total_start = time.time()

    for idx, row in test_df.iterrows():
        image_path = row['full_path']
        true_label = row['true_label']

        elapsed, pred_label = infer_image(image_path)
        
        results.append({
            "image_order": idx,
            "elapsed_time": elapsed,
            "pred_label": pred_label,
            "true_label": true_label
        })

        if idx % 1000 == 0:
            print(f"✅ Processed {idx} images...")

    # --- End total timing ---
    total_end = time.time()
    total_elapsed = total_end - total_start

    print(f"✅ Total sequential inference time: {total_elapsed:.2f} seconds.")

    # --- Save results ---
    results_df = pd.DataFrame(results)
    os.makedirs("inference_tests/sequential/csv", exist_ok=True)
    results_df.to_csv("inference_tests/sequential/csv/sequential_inference_results.csv", index=False)
    print("✅ Saved sequential inference results to inference_tests/sequential/csv/sequential_inference_results.csv")
