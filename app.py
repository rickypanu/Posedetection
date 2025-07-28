# app.py
import tempfile

import cv2
import mediapipe as mp
import streamlit as st

from pose_module import process_frame

st.title("Real-time Pose Detection")

# Setup pose detector
mp_pose = mp.solutions.pose

# Sidebar input
option = st.sidebar.selectbox("Select Input", ["Webcam", "Upload Video"])

# Set up video source
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
    else:
        cap = None
else:
    cap = cv2.VideoCapture(0)

# Streamlit video frame placeholder
if cap:
    stframe = st.empty()
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = process_frame(frame, pose)
            stframe.image(image, channels="BGR", use_container_width=True)  # ✅ Updated line

    cap.release()

