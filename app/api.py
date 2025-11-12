from flask import Flask, request, jsonify
from flask_cors import CORS
from app.model_utils import predict_emotion, grad_cam

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field 'file'."}), 400
    file = request.files["file"]
    try:
        result = predict_emotion(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/predict-with-cam")
def predict_with_cam():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field 'file'."}), 400
    file = request.files["file"]
    try:
        b64 = grad_cam(file.read())
        result = predict_emotion(file.read())
        result["grad_cam_png_b64"] = b64
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)