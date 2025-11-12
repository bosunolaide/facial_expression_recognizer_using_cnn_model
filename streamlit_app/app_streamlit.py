import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://localhost:5000")

st.set_page_config(page_title="😃 Facial Expression Recognizer", page_icon="😃", layout="centered")
st.title("😃 Facial Expression Recognition (CNN)")
st.write("Upload an image of a face to classify the emotion.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

use_cam = st.toggle("Show Grad-CAM visualization")

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)

    files = {"file": uploaded.getvalue()}
    endpoint = "/predict-with-cam" if use_cam else "/predict"

    with st.spinner("Analyzing..."):
        try:
            resp = requests.post(f"{API_URL}{endpoint}", files=files, timeout=60)
            if not resp.ok:
                st.error(resp.text)
            else:
                data = resp.json()
                st.success(f"**Emotion:** {data['emotion'].title()}  |  Confidence: {data['confidence']*100:.2f}%")
                if use_cam and "grad_cam_png_b64" in data:
                    cam = Image.open(io.BytesIO(base64.b64decode(data["grad_cam_png_b64"])))
                    st.image(cam, caption="Grad-CAM", use_column_width=True)
        except Exception as e:
            st.error(f"Request failed: {e}")