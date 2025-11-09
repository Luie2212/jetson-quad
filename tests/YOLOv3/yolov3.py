#!/usr/bin/env python3.8
"""
Standalone Python3 camera capture loop
Works on Windows and Linux without ROS
"""

import time
import cv2
import os
from ultralytics import YOLO

if __name__ == '__main__':
    # # define resolution
    # resCap = (1920, 1080)
    # resWrite = resCap

    # # open device (0 = default webcam on Windows)
    # cap = cv2.VideoCapture("/dev/video0")
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, resCap[0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resCap[1])
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # if not cap.isOpened():
    #     print("Error: Could not open camera.")
    #     exit()

    # Load a YOLOv3 model (e.g., yolov3, yolov3-spp)
    model =YOLO("tests/YOLOv3/yolov3u.pt")  # specify 'yolov3' or other variants

    # make sure save folder exists
    save_dir_raw = os.path.join(os.getcwd(), "tests/YOLOv3/Photos/Raw")
    os.makedirs(save_dir_raw, exist_ok=True)
    save_dir_inf = os.path.join(os.getcwd(), "tests/YOLOv3/Photos/Infered")
    os.makedirs(save_dir_inf, exist_ok=True)

    image_num = 0
    capture_interval = 5.0  # seconds between captures
    prev_time = time.time()

    # print("Starting camera capture loop... press Ctrl+C to exit.")

    try:
        while True:
            # ret, frame = cap.read()
            
            ret = True
            frame = cv2.imread(f"tests/YOLOv3/test.jpeg")

            if ret:
                # Save the captured photo
                filename = f'test.jpeg'
                filepath = os.path.join(save_dir_raw, filename)
                cv2.imwrite(filepath, frame)
                print(f"Saved Image as {filepath}")
            else:
                print("Failed to capture photo.")
            
            # Perform yolo inference
            results = model(frame)

            # Draw bounding boxes directly on the frame
            annotated_frame = results[0].plot()

            # # Show in a live OpenCV window
            # cv2.imshow("YOLO Camera", annotated_frame)
            # Save the captured photo
            filename = f'test_classified.jpeg'
            filepath = os.path.join(save_dir_inf, filename)
            cv2.imwrite(filepath, annotated_frame)
            print(f"Saved Image as {filepath}")

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
