# ==========================================================
# Uses HSV + LAB + adaptive normalization for robust color separation
# Includes grid search for optimal SVM parameters
# ==========================================================

import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, classification_report
)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import joblib
from collections import Counter

# ==========================================================
# Dataset Paths
# ==========================================================
ORIGINAL_DIR = r"D:\Yams\College\Sem_5\Project_main\proper\dataset\input_dataset1"
AUGMENTED_DIR = r"D:\Yams\College\Sem_5\Project_main\proper\dataset\input_dataset2"
USER_DIR = r"D:\Yams\College\Sem_5\Project_main\proper\user_dataset"

MODEL_PATH = r"D:\Yams\College\Sem_5\Project_main\proper\model"
os.makedirs(MODEL_PATH, exist_ok=True)

COLORS = ["white", "orange", "yellow", "red", "blue", "green"]

# ==========================================================
# Adaptive HSV Normalization
# ==========================================================
def normalize_hsv(arr):
    hsv = np.array(arr, dtype=np.float32).reshape(-1, 3)
    h = hsv[:, 0].reshape(-1)
    s = hsv[:, 1].reshape(-1)
    v = hsv[:, 2].reshape(-1)

    # Adjust brightness and saturation globally
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).reshape(-1)
    s = np.clip(s, 50, 255).reshape(-1)

    hsv[:, 0] = h
    hsv[:, 1] = s
    hsv[:, 2] = v
    return hsv

# ==========================================================
# HSV Augmentation
# ==========================================================
def augment_hsv(arr, color):
    arr = arr.copy()
    n = arr.shape[0]
    if color == "orange":
        arr[:, 0] = np.clip(arr[:, 0] - np.random.uniform(3, 6, n), 0, 179)
    elif color == "yellow":
        arr[:, 0] = np.clip(arr[:, 0] + np.random.uniform(3, 6, n), 0, 179)
    elif color == "green":
        arr[:, 0] = np.clip(arr[:, 0] + np.random.uniform(-2, 2, n), 0, 179)
    return arr

# ==========================================================
# Feature Extraction (HSV + LAB)
# ==========================================================
def extract_combined_features(hsv_arr):
    """Convert HSV to RGB→LAB and combine HSV + LAB"""
    rgb_arr = cv2.cvtColor(np.uint8([hsv_arr]), cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    lab_arr = cv2.cvtColor(np.uint8([rgb_arr]), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    combined = np.hstack([hsv_arr, lab_arr])  # Shape: (N, 6)
    return combined

# ==========================================================
# Load Dataset (Robust for 1D/2D arrays)
# ==========================================================
def load_dataset(base_dir):
    X_list, y_list = [], []
    for color in COLORS:
        color_dir = os.path.join(base_dir, color)
        if not os.path.exists(color_dir):
            continue

        for fname in os.listdir(color_dir):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(color_dir, fname))

                arr = np.atleast_2d(arr)
                if arr.shape[1] != 3:
                    print(f"⚠️ Skipping {fname}: unexpected shape {arr.shape}")
                    continue

                n = arr.shape[0]
                arr = normalize_hsv(arr)

                # --- Apply augmentation for confusing colors ---
                if color in ["orange", "yellow", "green"]:
                    arr = augment_hsv(arr, color)

                # --- Minor tuning ---
                if color == "white":
                    arr[:, 2] *= np.random.uniform(0.95, 1.10)
                    arr[:, 1] *= 0.9
                elif color == "orange":
                    arr[:, 0] = np.clip(arr[:, 0] - np.random.uniform(2, 5, size=n), 0, 179)
                elif color == "yellow":
                    arr[:, 0] = np.clip(arr[:, 0] + np.random.uniform(2, 5, size=n), 0, 179)
                    arr[:, 1] = np.clip(arr[:, 1] * np.random.uniform(1.05, 1.15), 0, 255)
                elif color == "green":
                    arr[:, 1] *= np.random.uniform(1.05, 1.15)
                    arr[:, 2] *= np.random.uniform(0.9, 1.0)
                elif color == "red":
                    arr[:, 0] *= np.random.uniform(0.95, 1.05)
                elif color == "blue":
                    arr[:, 0] *= np.random.uniform(0.95, 1.05)

                # --- Extract combined HSV+LAB features ---
                feats = extract_combined_features(arr)
                X_list.append(feats)
                y_list.extend([color] * n)

    if not X_list:
        return np.empty((0, 6)), np.empty((0,))
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

# ==========================================================
# Load All Datasets (Original + Augmented + User)
# ==========================================================
print("\n--- Loading Datasets ---")

X_parts, y_parts = [], []
dataset_names = ["Original", "Augmented", "User"]
dataset_dirs = [ORIGINAL_DIR, AUGMENTED_DIR, USER_DIR]
counts = {}

for name, folder in zip(dataset_names, dataset_dirs):
    if os.path.exists(folder):
        X_temp, y_temp = load_dataset(folder)
        if X_temp.size > 0:
            X_parts.append(X_temp)
            y_parts.append(y_temp)
            counts[name] = len(y_temp)
        else:
            counts[name] = 0
    else:
        counts[name] = 0

if not X_parts:
    raise ValueError("No valid datasets found! Please check your dataset paths.")

X = np.vstack(X_parts)
y = np.concatenate(y_parts)

print(f"✅ Total Samples: {X.shape[0]}")
for name in dataset_names:
    print(f"   ├── {name:<10}: {counts[name]}")

# ==========================================================
# Shuffle, Encode, and Scale
# ==========================================================
X, y = shuffle(X, y, random_state=42)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# Train/Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# ==========================================================
# Train SVM (with Grid Search)
# ==========================================================
param_grid = {
    "C": [5, 10, 15, 20, 25],
    "gamma": [0.1, 0.05, 0.01, 0.005],
    "kernel": ["rbf"]
}

print("\n🔍 Running GridSearchCV for best SVM parameters...")
svm = SVC(class_weight="balanced")
grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

svm_best = grid.best_estimator_
print(f"\n✅ Best Parameters: {grid.best_params_}\n")

# ==========================================================
# Evaluate
# ==========================================================
y_train_pred = svm_best.predict(X_train)
y_test_pred = svm_best.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"✅ Training Accuracy: {train_acc:.2f}%")
print(f"✅ Testing Accuracy:  {test_acc:.2f}%\n")

print("📊 Classification Report (Test Data):\n")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# ==========================================================
# Confusion Matrices
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

ConfusionMatrixDisplay(cm_train, display_labels=le.classes_).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Training Confusion Matrix ({train_acc:.2f}%)")
axes[0].tick_params(axis='x', rotation=45)

ConfusionMatrixDisplay(cm_test, display_labels=le.classes_).plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title(f"Testing Confusion Matrix ({test_acc:.2f}%)")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ==========================================================
# Save Model
# ==========================================================
joblib.dump(svm_best, os.path.join(MODEL_PATH, "svm_rbf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
joblib.dump(le, os.path.join(MODEL_PATH, "label_encoder.pkl"))

print(f"\n✅ Model, scaler, and label encoder saved successfully to: {MODEL_PATH}\n")

# ==========================================================
# Class Summary
# ==========================================================
print("--- Class Distribution ---")
counts = Counter(y)
for c, cnt in counts.items():
    print(f"{c:<8}: {cnt}")