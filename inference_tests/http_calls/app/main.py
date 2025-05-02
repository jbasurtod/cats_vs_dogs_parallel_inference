# --- 📦 Imports ---
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from worker import infer_image

# --- 📂 Load CSV with images ---
csv_path = "/app/csv/training_images.csv"
df = pd.read_csv(csv_path)

# --- 🧹 Filter test images only ---
test_df = df[df['type'] == 'test'].reset_index(drop=True)
image_list = test_df['image'].tolist()

# --- 📂 Ensure results folder exists ---
os.makedirs("/app/results", exist_ok=True)

print(f"✅ Total test images to infer: {len(image_list)}")

# --- 🚀 Function to save result ---
def save_result(result):
    api_id = result.get("api_id", 0)
    result_file = f"/app/results/{api_id}_results.csv"

    result_df = pd.DataFrame([result])

    if os.path.exists(result_file):
        result_df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(result_file, index=False)

# --- 🚀 Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Define how many concurrent threads you want
    num_threads = 6

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Launch all requests in parallel
        future_to_image = {executor.submit(infer_image, img): img for img in image_list}

        for idx, future in enumerate(as_completed(future_to_image)):
            result = future.result()
            save_result(result)

            if idx % 500 == 0:
                print(f"✅ Processed {idx} images...")

    total_time = time.time() - start_time
    print(f"✅ All inferences completed in {total_time:.2f} seconds.")
