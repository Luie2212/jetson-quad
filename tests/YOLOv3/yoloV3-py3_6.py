#!/usr/bin/env python3
import os
import time
import cv2
import torch
import numpy as np
import sys

repo_path = os.path.expanduser("~/jetson-quad/tests/YOLOv3/PyTorch-YOLOv3/pytorchyolo")
sys.path.append(repo_path)
repo_path = os.path.expanduser("~/jetson-quad/tests/YOLOv3/PyTorch-YOLOv3")
sys.path.append(repo_path)

from models import Darknet
from utils.utils import non_max_suppression, rescale_boxes, load_classes

# ------------------ CONFIG ------------------
CFG_FILE = "config/yolov3.cfg"        # or yolov3-tiny.cfg
WEIGHTS = "yolov3.weights"            # or yolov3-tiny.weights
IMG_SIZE = 416
CONF_THRESH = 0.25
IOU_THRESH = 0.45
SAVE_DIR = os.path.expanduser("~/jetson-quad/tests/YOLOv3/Photos")
CAP_DEVICE = "/dev/video0"
CAPTURE_INTERVAL = 5.0
os.makedirs(SAVE_DIR, exist_ok=True)
# -------------------------------------------

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(CFG_FILE, img_size=IMG_SIZE).to(device)
model.load_darknet_weights(WEIGHTS)
model.eval()
classes = load_classes("data/coco.names")  # COCO classes

# Camera setup
cap = cv2.VideoCapture(CAP_DEVICE)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img

def draw_boxes(frame, detections):
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        label = f"{classes[int(cls_pred)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

# ------------------ CAPTURE LOOP ------------------
image_num = 0
next_time = time.time()

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        # Save original
        orig_path = os.path.join(SAVE_DIR, f"image_{image_num:04x}.jpg")
        cv2.imwrite(orig_path, frame)

        # Preprocess and run YOLO
        img_tensor = preprocess_frame(frame)
        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)

        # Rescale detections to original frame
        if pred[0] is not None:
            pred_rescaled = rescale_boxes(pred[0], IMG_SIZE, frame.shape[:2])
            annotated = draw_boxes(frame.copy(), pred_rescaled)
        else:
            annotated = frame.copy()

        # Save annotated image
        annotated_path = os.path.join(SAVE_DIR, f"image_{image_num:04x}_classified.jpg")
        cv2.imwrite(annotated_path, annotated)
        print(f"Saved {orig_path} and {annotated_path}")

        image_num += 1

        # --- timing control ---
        next_time += CAPTURE_INTERVAL
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # If inference took longer than CAPTURE_INTERVAL
            next_time = time.time()

except KeyboardInterrupt:
    print("Capture loop interrupted by user.")

cap.release()
cv2.destroyAllWindows()
