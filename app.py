import streamlit as st
from ultralytics import YOLO
import os
import shutil
import tempfile
import subprocess

model = YOLO('yolov8s.pt')
VEHICLE_CLASSES = [2, 3, 5, 7]

def process_video(input_video_path):
    if not input_video_path:
        return None
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_dir = os.path.join(base_dir, "runs", "detect")
    output_dir = os.path.join(project_dir, "tracking")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    results = model.track(
        source=input_video_path,
        save=True,
        project=project_dir,
        name="tracking",
        exist_ok=True,
        classes=VEHICLE_CLASSES,
        tracker="bytetrack.yaml",
        conf=0.25
    )
    
    for f in os.listdir(output_dir):
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = os.path.join(output_dir, f)
            # Convert to h264 for web playback if needed
            web_path = os.path.join(output_dir, "web_" + f)
            try:
                subprocess.run(['ffmpeg', '-y', '-i', output_path, '-vcodec', 'libx264', web_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return web_path
            except Exception:
                return output_path
            
    return None

st.set_page_config(page_title="Vehicle Tracking", page_icon="🚗")
st.title("🚗 Vehicle Object Detection & Tracking")
st.markdown("Upload a video of traffic or vehicles to automatically detect and track them in real-time. Powered by **YOLOv8** for accurate detection and **ByteTrack** for high-speed object tracking.")

uploaded_file = st.file_uploader("Upload Input Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.video(tfile.name)
    
    if st.button("Detect & Track Vehicles"):
        with st.spinner("Processing video (this may take a few minutes depending on the video length)..."):
            output_video_path = process_video(tfile.name)
            
        if output_video_path:
            st.success("Processing complete!")
            st.video(output_video_path)
        else:
            st.error("Failed to process video.")
