import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

LABEL_NAMES = [
    "BACKGROUND", "SKIN", "NOSE", "RIGHT_EYE", "LEFT_EYE", "RIGHT_BROW", "LEFT_BROW",
    "RIGHT_EAR", "LEFT_EAR", "MOUTH_INTERIOR", "TOP_LIP", "BOTTOM_LIP", "NECK",
    "HAIR", "BEARD", "CLOTHING", "GLASSES", "HEADWEAR", "FACEWEAR"
]

IGNORE_INDEX = 255
NUM_CLASSES = len(LABEL_NAMES)

def load_label_paths(label_dir, extensions={".png", ".jpg"}):
    paths = []
    for root, _, files in os.walk(label_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in extensions:
                paths.append(os.path.join(root, f))
    return sorted(paths)

def compute_class_weights_efficient(label_paths, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for path in tqdm(label_paths, desc="Counting pixels"):
        label = np.array(Image.open(path))
        mask = (label != ignore_index)
        filtered = label[mask]
        
        if np.any(filtered >= num_classes):
            invalid_values = np.unique(filtered[filtered >= num_classes])
            print(
                f"WARNING: label file {path} contains illegal class values ??{invalid_values} "
                f" (allowed range 0-{num_classes-1}), automatically truncated"
            )
            filtered = filtered[filtered < num_classes] 

        hist = np.bincount(filtered.flatten(), minlength=num_classes)
        class_counts += hist
        

    total = class_counts.sum()
    weights = total / (class_counts * num_classes)
    return torch.tensor(weights, dtype=torch.float32)

def save_outputs(weights, output_dir="weights_output"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. CSV
    df = pd.DataFrame({
        "ClassID": list(range(NUM_CLASSES)),
        "Label": LABEL_NAMES,
        "Weight": weights.numpy()
    })
    df.to_csv(os.path.join(output_dir, "class_weights.csv"), index=False)

    # 2. Torch tensor
    torch.save(weights, os.path.join(output_dir, "class_weights.pt"))

    # 3. Plot
    plt.figure(figsize=(12, 5))
    plt.bar(LABEL_NAMES, weights.numpy())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Class Weight")
    plt.title("Class Weights by Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_weights.png"))
    plt.close()

    print(f"\n? Saved:\n- CSV: class_weights.csv\n- .pt: class_weights.pt\n- Image: class_weights.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", required=True, help="Directory with label images")
    parser.add_argument("--output_dir", default="weights_output", help="Output folder for weights")
    args = parser.parse_args()

    label_paths = load_label_paths(args.label_dir)
    weights = compute_class_weights_efficient(label_paths)
    save_outputs(weights, args.output_dir)
