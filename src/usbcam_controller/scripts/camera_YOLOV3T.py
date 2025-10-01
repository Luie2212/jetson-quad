#!/usr/bin/python3.8
"""
Standalone Python3 camera capture loop
Works on Windows and Linux without ROS
"""

import time
import cv2
import os
from ultralytics import YOLO

def get_ros_time():
    """Read the latest ROS time from the file, fallback to system time if unavailable."""
    try:
        with open("/tmp/ros_time.txt", "r") as f:
            return f.readline().strip()
    except Exception:
        return str(time.time())

if __name__ == '__main__':
    # define resolution
    resCap = (1920, 1080)
    resWrite = resCap

    # open device (0 = default webcam on Windows)
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resCap[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resCap[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Load a YOLOv3 model (e.g., yolov3, yolov3-spp)
    model =YOLO("yolov3-tinyu.pt")  # specify 'yolov3' or other variants

    # make sure save folder exists
    save_dir_raw = os.path.join(os.getcwd(), "Photos/Raw")
    os.makedirs(save_dir_raw, exist_ok=True)
    save_dir_inf = os.path.join(os.getcwd(), "Photos/Infered")
    os.makedirs(save_dir_inf, exist_ok=True)

    image_num = 0
    capture_interval = 5.0  # seconds between captures
    prev_time = time.time()

    print("Starting camera capture loop... press Ctrl+C to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # Save the captured photo
                ros_time = get_ros_time()
                filename_raw = f'image_{image_num:04x}_{ros_time}.jpg'
                filepath_raw = os.path.join(save_dir_raw, filename_raw)
                cv2.imwrite(filepath_raw, frame)
                print(f"Saved Image as {filepath_raw}")
            else:
                print("Failed to capture photo.")
                continue
            
            # Perform yolo inference
            results = model(frame)

            # Draw bounding boxes directly on the frame
            annotated_frame = results[0].plot()

            # # Show in a live OpenCV window
            # cv2.imshow("YOLO Camera", annotated_frame)
            # Save the captured photo
            filename_inf = f'image_{image_num:04x}_{ros_time}_classified.jpg'
            filepath_inf = os.path.join(save_dir_inf, filename_inf)
            cv2.imwrite(filepath_inf, annotated_frame)

            image_num += 1

            # time tracking
            curr_time = time.time()
            elapsed = curr_time - prev_time
            print(f"Capture-inference loop took {elapsed:.4f} seconds")
            prev_time = curr_time

            # sleep only if processing was faster than interval
            remaining = capture_interval - elapsed
            print(f"Waiting {elapsed:.4f} + {remaining:.4f} for a total of {(elapsed + remaining):.4f} seconds")
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("Capture loop interrupted by user.")

    cap.release()
    cv2.destroyAllWindows()
