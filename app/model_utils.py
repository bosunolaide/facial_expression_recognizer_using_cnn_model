import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import base64
from typing import Tuple, Dict

MODEL_PATH = os.getenv("MODEL_PATH", "model/FacialExpressionModel.h5")
ENCODER_PATH = os.getenv("ENCODER_PATH", "model/LabelEncoder.pck")

_MODEL = None
_ENCODER = None

def load_model_and_encoder():
    global _MODEL, _ENCODER
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    if _ENCODER is None:
        with open(ENCODER_PATH, "rb") as f:
            _ENCODER = pickle.load(f)
    return _MODEL, _ENCODER

def preprocess_image_bytes(image_bytes: bytes, target_size: Tuple[int,int]=(48,48)) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image bytes")
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def predict_emotion(image_bytes: bytes) -> Dict:
    model, encoder = load_model_and_encoder()
    x = preprocess_image_bytes(image_bytes)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    emotion = encoder.inverse_transform([idx])[0] if hasattr(encoder, "inverse_transform") else str(idx)
    confidence = float(preds[idx])
    return {"emotion": emotion, "confidence": confidence, "probs": preds.tolist()}

def grad_cam(image_bytes: bytes, last_conv_layer_name: str=None, alpha: float=0.4):
    model, encoder = load_model_and_encoder()
    # pick last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found")
    x = preprocess_image_bytes(image_bytes)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    # original
    nparr = np.frombuffer(image_bytes, np.uint8)
    orig = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    orig = cv2.resize(orig, (48,48))
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    heatmap_resized = cv2.resize(heatmap, (orig_rgb.shape[1], orig_rgb.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(heatmap_colored, alpha, orig_rgb, 1 - alpha, 0)
    _, buf = cv2.imencode(".png", superimposed)
    b64 = base64.b64encode(buf).decode("utf-8")
    return b64