import cv2
import torch
import numpy as np
import os

print("PyTorch version:", torch.__version__)

# --- Model Setup ---
model_type = "MiDaS_small"
print("Loading MiDaS model:", model_type)
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# --- Configuration ---
# 1. Set this to the main folder that contains your subfolders
DATASET_BASE_FOLDER = "OutputFrames"

# 2. Map your exact folder names to the required output strings.
# Change the keys (e.g., "left_frames") if your folders are named differently.
FOLDER_TO_LABEL_MAP = {
    "left": "Go left",
    "center": "Go straight",
    "right": "Go right",
    "stop": "Stop obstacle ahead"
}

OBSTACLE_THRESHOLD = 250

def predict_frame(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Depth Prediction
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # Image Segmentation mapped to your desired outputs
    img_width = img_rgb.shape[1]
    segments = {
        "Go left": (0, int(0.33 * img_width)),
        "Go straight": (int(0.33 * img_width), int(0.66 * img_width)),
        "Go right": (int(0.66 * img_width), img_width)
    }

    # Analyze Segments
    segment_avg_depths = {}
    for name, (start, end) in segments.items():
        segment_depth = depth_map[:, start:end]
        avg_depth = np.mean(segment_depth)
        segment_avg_depths[name] = avg_depth

    clearest_segment_name = min(segment_avg_depths, key=segment_avg_depths.get)
    clearest_segment_depth_value = segment_avg_depths[clearest_segment_name]

    if clearest_segment_depth_value > OBSTACLE_THRESHOLD:
        return "Stop obstacle ahead"
    else:
        return clearest_segment_name

# --- Folder-Based Evaluation Loop ---
def run_evaluation():
    if not os.path.exists(DATASET_BASE_FOLDER):
        print(f"Error: The base folder '{DATASET_BASE_FOLDER}' does not exist.")
        return

    overall_correct = 0
    overall_total = 0
    class_stats = {label: {"correct": 0, "total": 0} for label in FOLDER_TO_LABEL_MAP.values()}

    for folder_name, true_label in FOLDER_TO_LABEL_MAP.items():
        folder_path = os.path.join(DATASET_BASE_FOLDER, folder_name)

        if not os.path.exists(folder_path):
            print(f"Skipping: Folder '{folder_name}' not found in {DATASET_BASE_FOLDER}.")
            continue

        print(f"\nEvaluating folder: '{folder_name}' (Target: {true_label})...")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

        for filename in images:
            img_path = os.path.join(folder_path, filename)

            predicted_label = predict_frame(img_path)
            if predicted_label is None:
                print(f"  Failed to read image: {filename}")
                continue

            is_correct = (predicted_label == true_label)

            # Update metrics
            overall_total += 1
            class_stats[true_label]["total"] += 1
            if is_correct:
                overall_correct += 1
                class_stats[true_label]["correct"] += 1

            # Optional: Comment out the line below if you don't want to see every single frame printed
            print(f"  File: {filename} | Pred: '{predicted_label}' | Correct: {is_correct}")

    # --- Print Final Results ---
    if overall_total > 0:
        print("\n" + "="*45)
        print("EVALUATION RESULTS")
        print("="*45)
        print(f"Total Frames Tested: {overall_total}")
        print(f"Correct Predictions: {overall_correct}")
        print(f"Overall Accuracy:    {(overall_correct / overall_total) * 100:.2f}%\n")

        print("--- Breakdown by Class ---")
        for label, stats in class_stats.items():
            if stats["total"] > 0:
                acc = (stats["correct"] / stats["total"]) * 100
                print(f"{label.ljust(20)}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
            else:
                print(f"{label.ljust(20)}: No images tested.")
        print("="*45)
    else:
        print("\nNo valid predictions were made. Check your folder paths and image formats.")

if __name__ == "__main__":
    run_evaluation()