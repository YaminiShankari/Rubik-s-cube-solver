# ==========================================================
# input_dataset2.py
# ----------------------------------------------------------
# Creates augmented dataset (~16–20% of total) for each color
# using HSV jittering based on verified input_dataset1 means.
# Saves 50 .npy files per color folder inside input_dataset2/
# Each color gets 450 samples total.
# ==========================================================

import numpy as np
import os

# ==========================================================
# 📘 Base mean HSV values (from verification1.py results)
# ==========================================================
BASE_HSV = {
    "white":  [111.07,  35.28, 138.84],
    "orange": [18.48,  172.18, 182.69],
    "yellow": [26.65,  179.77, 148.51],
    "red":    [171.78, 213.26, 217.83],
    "blue":   [104.99, 223.25, 179.49],
    "green":  [64.31,  175.89, 152.58]
}

# ==========================================================
# 🧠 Configuration
# ==========================================================
ORIGINAL_SAMPLES_PER_COLOR = 1800     # From input_dataset1 (10800 / 6)
AUGMENT_SAMPLES_PER_COLOR = 450       # ≈16–20% of original (84:16 ratio)
FILES_PER_COLOR = 50                  # Create 50 .npy files per color
OUTPUT_BASE = r"D:\Yams\College\Sem_5\Project_main\dataset\input_dataset2"

# ==========================================================
# 🎨 HSV Jitter Settings (per color)
# ==========================================================
VARIATION = {
    "white":  {"h": 5, "s": 10, "v": 25},
    "orange": {"h": 5, "s": 25, "v": 25},
    "yellow": {"h": 6, "s": 25, "v": 30},
    "red":    {"h": 5, "s": 20, "v": 25},
    "blue":   {"h": 6, "s": 20, "v": 25},
    "green":  {"h": 6, "s": 20, "v": 25},
}

# ==========================================================
# 🧩 Helper: Generate bounded random HSV samples
# ==========================================================
def generate_hsv_samples(mean_hsv, variation, n):
    mean = np.array(mean_hsv)
    stds = np.array([variation["h"], variation["s"], variation["v"]])
    samples = np.random.normal(loc=mean, scale=stds, size=(n, 3))

    # Clip to valid OpenCV HSV ranges
    samples[:, 0] = np.clip(samples[:, 0], 0, 179)
    samples[:, 1] = np.clip(samples[:, 1], 0, 255)
    samples[:, 2] = np.clip(samples[:, 2], 0, 255)

    # Special handling for white — low S, high V
    if mean_hsv == BASE_HSV["white"]:
        samples[:, 1] = np.clip(np.random.normal(25, 8, n), 0, 50)
        samples[:, 2] = np.clip(np.random.normal(235, 10, n), 210, 255)

    return samples.astype(np.float32)

# ==========================================================
# 🚀 Main Dataset Augmentation
# ==========================================================
def main():
    print("\n--- Rubik's Cube Dataset Augmentation (84:16 ratio) ---\n")

    total_augmented = 0

    for color, base_hsv in BASE_HSV.items():
        print(f"Generating augmented samples for {color}...")

        # Create color-specific directory
        color_dir = os.path.join(OUTPUT_BASE, color)
        os.makedirs(color_dir, exist_ok=True)

        # Generate all samples for this color
        total_samples = generate_hsv_samples(base_hsv, VARIATION[color], AUGMENT_SAMPLES_PER_COLOR)

        # Split into multiple .npy files
        split_size = AUGMENT_SAMPLES_PER_COLOR // FILES_PER_COLOR
        for i in range(FILES_PER_COLOR):
            start = i * split_size
            end = start + split_size if i < FILES_PER_COLOR - 1 else AUGMENT_SAMPLES_PER_COLOR
            np.save(os.path.join(color_dir, f"{color}_{i+1:02d}.npy"), total_samples[start:end])

        total_augmented += AUGMENT_SAMPLES_PER_COLOR
        print(f"✅ Created {FILES_PER_COLOR} .npy files for {color} ({AUGMENT_SAMPLES_PER_COLOR} samples total)")

    total_original = ORIGINAL_SAMPLES_PER_COLOR * len(BASE_HSV)
    total_all = total_original + total_augmented
    aug_ratio = (total_augmented / total_all) * 100

    print(f"\nTotals:\nOriginal samples total  : {total_original}")
    print(f"Augmented samples total : {total_augmented}")
    print(f"Original ratio: {100 - aug_ratio:.2f}%")
    print(f"Augmented ratio: {aug_ratio:.2f}%")
    print(f"\n🎉 Augmented dataset created successfully in: {OUTPUT_BASE}")

# ==========================================================
# 🏁 Run
# ==========================================================
if __name__ == "__main__":
    main()