# ==========================================================
# Rubik's Cube Color Prediction + Adaptive Retraining
# ==========================================================

import cv2
import numpy as np
import joblib
import os
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
MODEL_PATH = r"D:\Yams\College\Sem_5\Project_main\proper\model"
USER_DATASET_PATH = r"D:\Yams\College\Sem_5\Project_main\proper\user_dataset"
os.makedirs(USER_DATASET_PATH, exist_ok=True)

svm_model = joblib.load(os.path.join(MODEL_PATH, "svm_rbf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

COLORS = ["white", "orange", "yellow", "red", "blue", "green"]

# ----------------------------------------------------------
# Face order and guidance
# ----------------------------------------------------------
FACES = [
    ("white", "Place the WHITE centered face."),
    ("orange", "Now show the ORANGE centered face."),
    ("yellow", "Now show the YELLOW centered face."),
    ("red", "Now show the RED centered face."),
    ("blue", "Now show the BLUE centered face."),
    ("green", "Now show the GREEN centered face.")
]

# ----------------------------------------------------------
# Adaptive HSV Normalization
# ----------------------------------------------------------
def normalize_hsv(arr):
    hsv = np.array(arr, dtype=np.float32).reshape(-1, 3)
    h = hsv[:, 0].reshape(-1)
    s = hsv[:, 1].reshape(-1)
    v = hsv[:, 2].reshape(-1)

    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).reshape(-1)
    s = np.clip(s, 50, 255).reshape(-1)

    hsv[:, 0] = h
    hsv[:, 1] = s
    hsv[:, 2] = v
    return hsv

# ----------------------------------------------------------
# HSV + LAB Feature Extraction
# ----------------------------------------------------------
def extract_combined_features(hsv_arr):
    rgb_arr = cv2.cvtColor(np.uint8([hsv_arr]), cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    lab_arr = cv2.cvtColor(np.uint8([rgb_arr]), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    return np.hstack([hsv_arr, lab_arr])

# ----------------------------------------------------------
# Draw 3x3 Grid
# ----------------------------------------------------------
def draw_grid(frame, top_left, size):
    x, y = top_left
    cell = size // 3
    cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 2)
    for i in range(1, 3):
        cv2.line(frame, (x + i * cell, y), (x + i * cell, y + size), (255, 255, 255), 2)
        cv2.line(frame, (x, y + i * cell), (x + size, y + i * cell), (255, 255, 255), 2)

# ----------------------------------------------------------
# Get HSV means for 9 cells
# ----------------------------------------------------------
def extract_9_hsv(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    hsv_values = []
    for r in range(3):
        for c in range(3):
            y1, x1 = r * cell_h, c * cell_w
            cell = hsv[y1:y1 + cell_h, x1:x1 + cell_w]
            mean_h, mean_s, mean_v = np.mean(cell[:, :, 0]), np.mean(cell[:, :, 1]), np.mean(cell[:, :, 2])
            hsv_values.append([mean_h, mean_s, mean_v])
    return np.array(hsv_values, dtype=np.float32)

# ----------------------------------------------------------
# Predict colors
# ----------------------------------------------------------
def predict_colors(hsv_values):
    hsv_norm = normalize_hsv(hsv_values)
    features = extract_combined_features(hsv_norm)
    scaled = scaler.transform(features)
    preds = svm_model.predict(scaled)
    decoded = label_encoder.inverse_transform(preds)
    return decoded.reshape(3, 3)

# ----------------------------------------------------------
# Save user samples
# ----------------------------------------------------------
def save_user_samples(predicted_colors, hsv_values):
    for color, hsv in zip(predicted_colors.flatten(), hsv_values):
        save_dir = os.path.join(USER_DATASET_PATH, color)
        os.makedirs(save_dir, exist_ok=True)
        file_index = len(os.listdir(save_dir)) + 1
        np.save(os.path.join(save_dir, f"{color}_{file_index}.npy"), hsv)

# ----------------------------------------------------------
# Retrain model with new user samples
# ----------------------------------------------------------
def retrain_model():
    X_list, y_list = [], []
    for color in os.listdir(USER_DATASET_PATH):
        color_dir = os.path.join(USER_DATASET_PATH, color)
        if not os.path.isdir(color_dir): continue
        for file in os.listdir(color_dir):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(color_dir, file))
                arr = np.atleast_2d(arr)
                if arr.shape[1] == 3:
                    X_list.append(arr)
                    y_list.extend([color] * arr.shape[0])

    if not X_list:
        print("⚠️ No new user data to retrain.")
        return

    X = np.vstack(X_list)
    y = np.array(y_list)
    X = normalize_hsv(X)
    X = extract_combined_features(X)
    X, y = shuffle(X, y, random_state=42)

    y_enc = label_encoder.fit_transform(y)
    X_scaled = scaler.fit_transform(X)

    new_model = SVC(kernel="rbf", C=15, gamma="scale", class_weight="balanced")
    new_model.fit(X_scaled, y_enc)

    joblib.dump(new_model, os.path.join(MODEL_PATH, "svm_rbf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))
    print("\n✅ Model retrained successfully with user data!")

# ----------------------------------------------------------
# Verify color counts
# ----------------------------------------------------------
def check_color_counts(all_faces):
    flattened = [c for face in all_faces.values() for row in face for c in row]
    counts = Counter(flattened)
    print("\n--- Color Count Verification ---")
    for color, count in counts.items():
        print(f"{color:<8}: {count}")
    if all(count == 9 for count in counts.values()):
        print("\n✅ All colors appear 9 times! Proceeding...")
        return True
    print("\n❌ Color imbalance detected. Please re-capture.")
    return False

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    captured_faces = {}
    print("\n--- Rubik's Cube Face Capture ---")

    for face_color, instruction in FACES:
        print(f"\n🎯 Capture Face: {face_color.upper()}")
        print(instruction)
        print("Press 'c' to capture or 'q' to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera not accessible!")
                break

            frame_disp = cv2.flip(frame, 1)
            h, w = frame_disp.shape[:2]
            size = int(min(h, w) * 0.55)
            top_left = ((w - size) // 2, (h - size) // 2)

            draw_grid(frame_disp, top_left, size)
            cv2.putText(frame_disp, f"Center Face: {face_color.upper()}", (10, 30),
                        FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_disp, "Press 'c' to capture | 'q' to quit",
                        (10, h - 20), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("Rubik's Cube Capture", frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                roi = frame_disp[top_left[1]:top_left[1] + size, top_left[0]:top_left[0] + size]
                hsv_values = extract_9_hsv(roi)
                predicted = predict_colors(hsv_values)

                print("\nPredicted 3x3 Colors:")
                for row in predicted:
                    print(" ".join([f"{c[:3].upper():<5}" for c in row]))

                if input(f"✅ Confirm {face_color.upper()} face? (y/n): ").strip().lower() == 'y':
                    captured_faces[face_color] = predicted
                    save_user_samples(predicted, hsv_values)
                    print("🧠 Saved confirmed data for retraining!")
                    break
                else:
                    print("↩️ Re-capture this face.")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if check_color_counts(captured_faces):
        retrain_model()
        np.save("captured_faces.npy", captured_faces)
        print("✅ Cube data saved successfully!")

# ----------------------------------------------------------
if __name__ == "__main__":
    main()