# input_dataset1.py
# Usage: python input_dataset1.py
# Press 'c' to capture (save one sample for current color).
# Press 'q' to quit the whole program.

import cv2
import numpy as np
import os
from datetime import datetime

# --- Config ---
CAMERA_INDEX = 0
OUTPUT_DIR = "input_dataset1"
TARGET_PER_COLOR = 50 #change as per your requirements
# Colors / faces order (you can change order if needed)
COLORS = ["white", "orange", "yellow", "red", "blue", "green"]
# Size of ROI as fraction of smaller frame dimension
ROI_FRAC = 0.55
# Thickness for grid lines
GRID_THICKNESS = 2
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def draw_grid(frame, top_left, size):
    x, y = top_left
    cell = size // 3
    # outer rectangle
    cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 1)
    # vertical lines
    for i in range(1, 3):
        cx = x + i * cell
        cv2.line(frame, (cx, y), (cx, y + size), (255, 255, 255), GRID_THICKNESS)
    # horizontal lines
    for i in range(1, 3):
        cy = y + i * cell
        cv2.line(frame, (x, cy), (x + size, cy), (255, 255, 255), GRID_THICKNESS)

def get_cell_means_hsv(roi_bgr):
    """
    roi_bgr: numpy array of shape (size, size, 3) in BGR
    returns: numpy array shape (9,3) of mean HSV values for each cell in row-major order
    """
    h, w = roi_bgr.shape[:2]
    cell_h = h // 3
    cell_w = w // 3
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    means = []
    for r in range(3):
        for c in range(3):
            y1 = r * cell_h
            x1 = c * cell_w
            y2 = y1 + cell_h
            x2 = x1 + cell_w
            cell = hsv[y1:y2, x1:x2]
            # compute mean over H,S,V channels separately
            mean_h = float(np.mean(cell[:, :, 0]))
            mean_s = float(np.mean(cell[:, :, 1]))
            mean_v = float(np.mean(cell[:, :, 2]))
            means.append([mean_h, mean_s, mean_v])
    return np.array(means, dtype=np.float32)  # shape (9,3)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera. Check CAMERA_INDEX or webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        return

    height, width = frame.shape[:2]
    roi_size = int(min(height, width) * ROI_FRAC)
    # center the ROI
    top_left_x = (width - roi_size) // 2
    top_left_y = (height - roi_size) // 2

    # Create output directories
    ensure_dir(OUTPUT_DIR)
    for color in COLORS:
        ensure_dir(os.path.join(OUTPUT_DIR, color))

    print("\n--- Rubik's Cube Dataset Capture ---")
    print("Instructions:")
    print(" - Place the solved face of the cube inside the white 3x3 grid.")
    print(" - Press 'c' to capture and save one sample (saves mean HSV of 9 stickers).")
    print(" - Press 'q' at any time to quit.\n")

    for color in COLORS:
        print(f"=== Capturing color: {color} ===")
        print(f"Place the {color.upper()} face in the ROI. Need {TARGET_PER_COLOR} samples. Press 'c' to capture, 'q' to quit.\n")
        saved_count = len([n for n in os.listdir(os.path.join(OUTPUT_DIR, color)) if n.endswith(".npy")])
        print(f"Already saved samples for {color}: {saved_count}")

        while saved_count < TARGET_PER_COLOR:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame, trying again.")
                continue

            # Mirror frame for natural interaction
            frame_disp = cv2.flip(frame, 1)

            # Recompute top-left for mirrored frame (since we flipped, positions remain same visually)
            h, w = frame_disp.shape[:2]
            roi_size = int(min(h, w) * ROI_FRAC)
            top_left_x = (w - roi_size) // 2
            top_left_y = (h - roi_size) // 2
            draw_grid(frame_disp, (top_left_x, top_left_y), roi_size)

            # overlay instructions and status
            status_text = f"Color: {color}  Saved: {saved_count}/{TARGET_PER_COLOR}"
            cv2.putText(frame_disp, status_text, (10, 30), FONT, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS, cv2.LINE_AA)
            cv2.putText(frame_disp, "Press 'c' to capture, 'q' to quit program", (10, h - 20), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # small guide: draw small numbered boxes showing sticker numbers (optional)
            # We'll annotate sticker indices 1..9 at center of each cell for clarity
            cell = roi_size // 3
            idx = 1
            for r in range(3):
                for c in range(3):
                    cx = top_left_x + c * cell + cell // 2
                    cy = top_left_y + r * cell + cell // 2
                    cv2.putText(frame_disp, str(idx), (cx - 10, cy + 6), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    idx += 1

            cv2.imshow("Capture - Place solved face in grid", frame_disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # crop ROI from the original (not flipped) frame for consistency
                # because we flipped for display, flip back coordinates: take from flipped frame for simplicity
                flipped = cv2.flip(frame, 1)
                roi = flipped[top_left_y:top_left_y + roi_size, top_left_x:top_left_x + roi_size]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    print("Error: ROI has zero size, skipping capture.")
                    continue

                hsv_means = get_cell_means_hsv(roi)  # (9,3)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{color}_sample_{saved_count + 150:03d}.npy"
                filepath = os.path.join(OUTPUT_DIR, color, filename)
                try:
                    np.save(filepath, hsv_means)
                    saved_count += 1
                    print(f"[{color}] Saved {filename}  (shape: {hsv_means.shape})")
                except Exception as e:
                    print(f"Error saving file {filepath}: {e}")

            elif key == ord('q'):
                print("Quitting capture process early by user request.")
                cap.release()
                cv2.destroyAllWindows()
                print("Exiting.")
                return

        print(f"Completed {TARGET_PER_COLOR} samples for color: {color}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("All colors captured. Dataset saved in folder:", OUTPUT_DIR)

if __name__ == "__main__":

    main()

