# import os
# os.system("pip install opencv-python-headless")

import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model (Replace with your own trained model if needed)
model = YOLO("best.pt")  # or "yolov8n.pt" for the pre-trained model

# Streamlit UI
st.title("Real-Time Object Detection with YOLOv8")
st.sidebar.write("### Camera Options")

# Select camera index
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)

# Confidence threshold slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Open the webcam
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    st.error("Could not open the webcam")
    st.stop()

# Stream video
stframe = st.empty()  # Placeholder for video

while True:
    success, frame = cap.read()
    if not success:
        st.warning("Failed to capture image")
        break

    # Perform object detection
    results = model(frame)[0]

    # Draw bounding boxes on the frame
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
        conf = float(box.conf[0])  # Confidence score
        class_id = int(box.cls[0])  # Class index
        label = f"{model.names[class_id]}: {conf:.2%}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert frame to RGB (for Streamlit)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    # Display the frame in Streamlit
    stframe.image(img, use_column_width=True)

# Release webcam on exit
cap.release()
