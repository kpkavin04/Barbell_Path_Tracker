import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from tracker import track_point
from streamlit_drawable_canvas import st_canvas
from helper import color_name_to_bgr

st.set_page_config(page_title="Barbell Tracker", layout="wide")
st.title("Barbell Path Tracker")

st.divider()
st.subheader("Step 1: Upload a lifting video")
uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.divider()
    st.subheader("Step 2: Enter your preferred path color (e.g. red, green, blue, magenta, navy):")
    st.caption("pick a color that is not present in the video background")
    color_name = st.text_input("")
    color_bgr = color_name_to_bgr(color_name)

    #storage of uploadded video onto disk as OpenCV is unable to read videos from in-memory objects
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read()) 
    video_path = tfile.name

    #allows reading of each frame in the video uploaded
    cap = cv2.VideoCapture(video_path)
    isSuccessful, first_frame = cap.read()
    cap.release() #to prevent memory leaks after reading from the video

    if isSuccessful:
        #convert BGR to RGB for canvas
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_pil = Image.fromarray(first_frame_rgb)

        st.divider()
        st.subheader("Step 3: Click on the point to track (e.g. end of barbell)")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 1)",
            stroke_width=5,
            stroke_color="#00FF00",
            background_image=first_frame_pil,
            update_streamlit=True,
            height=first_frame.shape[0],
            width=first_frame.shape[1],
            drawing_mode="point",
            key="canvas",
        )

        if canvas_result.json_data and "objects" in canvas_result.json_data:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0 and st.button("Start Tracking"):
                pt = objects[0]
                x, y = int(pt["left"]), int(pt["top"])

                with st.spinner("Processing video..."):
                    output_path = os.path.join(tempfile.gettempdir(), "tracked_output.mp4")
                    result_path = track_point(video_path, (x, y), color_bgr, output_path)

                st.success("Done! Download below:")
                with open(result_path, "rb") as f:
                    st.download_button("Download video", f, file_name="barbell_tracked.mp4", mime="video/mp4")
