import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🚀 YOLOv8 Object Detection")
st.markdown("Upload an image or video to detect objects using AI!")

# Sidebar for model selection and settings
st.sidebar.title("Settings")

# Model selection
model_size = st.sidebar.selectbox(
    "Select Model Size",
    ["Nano (Fastest)", "Small", "Medium", "Large", "XLarge (Most Accurate)"]
)

# Confidence threshold
confidence = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5,
    help="Adjust how confident the model should be before showing detections"
)

# Load the selected model
@st.cache_resource
def load_model(model_size):
    model_map = {
        "Nano (Fastest)": "yolov8n.pt",
        "Small": "yolov8s.pt", 
        "Medium": "yolov8m.pt",
        "Large": "yolov8l.pt",
        "XLarge (Most Accurate)": "yolov8x.pt"
    }
    model = YOLO(model_map[model_size])
    return model

model = load_model(model_size)

# Main content area
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📹 Live Camera"])

# Tab 1: Image Detection
with tab1:
    st.header("Image Object Detection")
    
    uploaded_image = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # Display original image
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detected Objects")
            
            # Convert PIL to OpenCV format
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # Perform detection
            with st.spinner("Detecting objects..."):
                results = model(image_cv, conf=confidence)
                annotated_image = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.image(annotated_image_rgb, use_column_width=True)
                
                # Show detection summary
                st.subheader("Detection Summary")
                result = results[0]
                detected_objects = {}
                
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence_score = float(box.conf[0])
                        
                        if class_name in detected_objects:
                            detected_objects[class_name] += 1
                        else:
                            detected_objects[class_name] = 1
                    
                    # Display object counts
                    for obj_name, count in detected_objects.items():
                        st.write(f"🔹 **{obj_name}**: {count} detected")
                    
                    st.success(f"✅ Total objects detected: {len(result.boxes)}")
                else:
                    st.warning("No objects detected. Try lowering the confidence threshold.")

# Tab 2: Video Detection
with tab2:
    st.header("Video Object Detection")
    
    uploaded_video = st.file_uploader(
        "Choose a video...", 
        type=['mp4', 'avi', 'mov'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name
        
        st.subheader("Original Video")
        st.video(uploaded_video)
        
        if st.button("Process Video for Object Detection"):
            with st.spinner("Processing video... This may take a while depending on video length."):
                # Process video
                cap = cv2.VideoCapture(temp_video_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create temporary output file
                output_path = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Perform detection
                    results = model(frame, conf=confidence)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    processed_frames += 1
                    progress = processed_frames / total_frames
                    progress_bar.progress(progress)
                
                # Clean up
                cap.release()
                out.release()
                
                # Display processed video
                st.subheader("Processed Video with Object Detection")
                st.video(output_path)
                
                # Download button for processed video
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=file,
                        file_name="object_detection_video.mp4",
                        mime="video/mp4"
                    )
                
                # Cleanup temporary files
                os.unlink(temp_video_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)

# Tab 3: Live Camera
with tab3:
    st.header("Live Camera Object Detection")
    
    st.warning("⚠️ This feature requires camera access and may not work on all servers.")
    
    # Camera input
    camera_image = st.camera_input("Take a picture for object detection")
    
    if camera_image is not None:
        # Convert to OpenCV format
        image = Image.open(camera_image)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Perform detection
        with st.spinner("Detecting objects..."):
            results = model(image_cv, conf=confidence)
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("With Object Detection")
                st.image(annotated_image_rgb, use_column_width=True)
            
            # Show detection results
            result = results[0]
            if len(result.boxes) > 0:
                st.success(f"🎯 Detected {len(result.boxes)} objects!")
                
                detected_items = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence_score = float(box.conf[0])
                    detected_items.append(f"{class_name} ({confidence_score:.1%})")
                
                st.write("**Detected:**", ", ".join(detected_items))
            else:
                st.info("No objects detected. Try adjusting the confidence threshold.")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using [YOLOv8](https://github.com/ultralytics/ultralytics) | "
    "Powered by [Streamlit](https://streamlit.io)"
)