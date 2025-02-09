import streamlit as st
import numpy as np
from PIL import Image
from deepface import DeepFace
from mtcnn import MTCNN
import cv2

def main():
    st.set_page_config(page_title="PAN Card Face Verification", layout="wide")
    st.title("PAN Card Face Verification")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the mode", ["Upload PAN Card", "Capture Live Photo", "Verify Identity"])

    if app_mode == "Upload PAN Card":
        upload_pan_card()
    elif app_mode == "Capture Live Photo":
        capture_live_photo()
    elif app_mode == "Verify Identity":
        verify_identity()

def upload_pan_card():
    st.header("Step 1: Upload Your PAN Card Photo")
    pan_image_file = st.file_uploader("Upload your PAN card photo", type=["jpg", "jpeg", "png"])
    if pan_image_file is not None:
        pan_image = Image.open(pan_image_file)
        st.image(pan_image, caption="Uploaded PAN Card Photo")
        st.session_state['pan_image'] = pan_image
        st.success("PAN card image uploaded successfully.")
    else:
        st.warning("Please upload a PAN card image.")

def capture_live_photo():
    st.header("Step 2: Capture a Live Photo")
    live_image = st.camera_input("Capture a live photo")
    if live_image is not None:
        live_image = Image.open(live_image)
        st.image(live_image, caption="Captured Live Photo")
        st.session_state['live_image'] = live_image
        st.success("Live photo captured successfully.")
    else:
        st.warning("Please capture a live photo.")

def verify_identity():
    st.header("Step 3: Verify Identity")
    if 'pan_image' in st.session_state and 'live_image' in st.session_state:
        pan_image = np.array(st.session_state['pan_image'])
        live_image = np.array(st.session_state['live_image'])

        st.info("Processing...")

        try:
            result = DeepFace.verify(pan_image, live_image, model_name='VGG-Face', detector_backend='mtcnn', enforce_detection=False)
            if result["verified"]:
                st.success("Faces match! Identity verified.")
            else:
                st.error("Faces do not match. Identity verification failed.")
        except Exception as e:
            st.error(f"An error occurred during verification: {e}")
    else:
        st.warning("Please complete steps 1 and 2 before verification.")

if __name__ == "__main__":
    main()
