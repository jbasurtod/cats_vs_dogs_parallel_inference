# --- 📦 Imports ---
from pyspark.sql import SparkSession
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import os
import time
import random

# --- ⚙️ Spark configuration ---
spark = SparkSession.builder \
    .appName("PyTorch Spark Inference") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.task.cpus", "1") \
    .config("spark.sql.shuffle.partitions", "5") \
    .getOrCreate()

sc = spark.sparkContext

print(f"✅ Spark initialized with {sc.defaultParallelism} workers.")

# --- 🧠 PyTorch device configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# --- 📦 Load model function ---
def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load("/app/model/mobilenetv3_cats_vs_dogs.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# --- Load model once ---
model = load_model()

# --- 🚀 Single image inference function ---
def infer_with_timing(image_info):
    worker_id, image_order, image_path, true_label = image_info
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

    return (worker_id, image_order, elapsed, pred_label, true_label)

# --- 🚀 Main execution ---
if __name__ == "__main__":
    # --- Load CSV ---
    csv_path = "/app/csv/training_images.csv"
    df = pd.read_csv(csv_path)

    # --- Filter test images ---
    test_df = df[df['type'] == 'test']
    test_df = test_df.reset_index(drop=True)  # <<< Important fix
    print(f"✅ Total test images found: {len(test_df)}")


    # --- Map classes to numbers ---
    class_to_idx = {"cat": 0, "dog": 1}
    test_df['true_label'] = test_df['class'].map(class_to_idx)

    images_info = []
    for idx, row in test_df.iterrows():
        image_path = os.path.join('/', row['image']) if not row['image'].startswith('/') else row['image']
        true_label = row['true_label']
        images_info.append((idx, image_path, true_label))


    # --- Shuffle images to improve parallelism ---
    random.shuffle(images_info)

    print(f"✅ Ready to infer {len(images_info)} images after shuffling.")

    # --- Prepare data with (worker_id, image_order, path, true_label) ---
    num_partitions = 5
    rdd = sc.parallelize(images_info, numSlices=num_partitions)

    def prepare_worker_data(index, iterator):
        for image_order, path, true_label in iterator:
            yield (index, image_order, path, true_label)

    worker_rdd = rdd.mapPartitionsWithIndex(prepare_worker_data)

    # --- Start timing total inference ---
    global_start = time.time()

    # --- Inference with timing ---
    results = worker_rdd.map(infer_with_timing).collect()

    # --- End timing total inference ---
    global_end = time.time()
    total_elapsed = global_end - global_start

    print(f"✅ Total inference time: {total_elapsed:.2f} seconds.")

    # --- Organize results into DataFrame ---
    results_df = pd.DataFrame(results, columns=["worker_id", "image_order", "elapsed_time", "pred_label", "true_label"])

    # --- Save detailed inference results ---
    os.makedirs("/app/csv", exist_ok=True)
    results_df.to_csv("/app/csv/inference_results.csv", index=False)
    print("✅ Saved detailed inference results to /app/csv/inference_results.csv")

    # --- Calculate worker statistics ---
    worker_stats = results_df.groupby("worker_id")["elapsed_time"].agg(["mean", "min", "max"]).reset_index()
    worker_stats.to_csv("/app/csv/worker_stats.csv", index=False)
    print("✅ Saved worker timing stats to /app/csv/worker_stats.csv")

    # --- Stop Spark ---
    spark.stop()
    print("✅ Inference process finished.")
