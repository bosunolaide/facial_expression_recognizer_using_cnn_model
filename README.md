![CI](https://github.com/bosunolaide/facial-expression-recognizer/actions/workflows/ci.yml/badge.svg)

# 😃 Facial Expression Recognition (CNN) — Full-Stack, Dockerized

Production-style implementation of a **facial expression recognizer** using a **Convolutional Neural Network (CNN)**. 
Includes a **Flask REST API** for inference, a **Streamlit frontend** for demos, **Grad‑CAM** visualizations, and a **Dockerfile** to run both services in one container.

## ✨ Highlights
- **Deep Learning**: Keras/TensorFlow CNN trained on 48×48 grayscale faces.
- **REST API**: `POST /predict` returns emotion + confidence; `POST /predict-with-cam` returns Grad‑CAM overlay.
- **Frontend**: Streamlit UI to upload images and view predictions/heatmaps.
- **Dockerized**: One container, two services (Flask on 5000, Streamlit on 8501).
- **Reproducible**: Pinned requirements and packaged model/encoder.

## 📦 Project Structure
```
facial_expression_recognizer/
├── app/
│   ├── api.py                # Flask routes
│   └── model_utils.py        # Preprocess, predict, Grad‑CAM
├── streamlit_app/
│   └── app_streamlit.py      # Streamlit UI
├── model/
│   ├── FacialExpressionModel.h5
│   ├── best_weights.h5
│   └── LabelEncoder.pck
├── notebooks/                # Original notebooks
├── assets/                   # (optional) screenshots / metrics
├── Dockerfile
├── requirements.txt
├── run.sh
└── README.md
```

## 🎥 Demo Preview

Below is a preview of the Streamlit app detecting emotions in real-time.
(Replace `assets/demo.gif` with your own recorded demo GIF.)

![Demo](assets/demo.gif)


## 🚀 Quickstart
### Docker
```bash
docker build -t facial-expr-app .
docker run -p 5000:5000 -p 8501:8501 facial-expr-app
```
- API health: `http://localhost:5000/health`
- UI: `http://localhost:8501`

### Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

## 🔌 REST API
### `POST /predict`
- field: `file` (image bytes)
- response: `{ "emotion": "happy", "confidence": 0.94, "probs": [...] }`

### `POST /predict-with-cam`
- field: `file`
- response: above + `grad_cam_png_b64` (Base64 PNG)

## 🧠 Model Notes
- Input: 48×48 grayscale
- Typical architecture: Conv → ReLU → Pool → Conv → ReLU → Pool → Dense → Dropout → Softmax
- Print summary:
```python
import tensorflow as tf; m = tf.keras.models.load_model("model/FacialExpressionModel.h5"); m.summary()
```

## 🧪 Tests
See `tests/test_api_basic.py` for basic route checks.

## License
MIT