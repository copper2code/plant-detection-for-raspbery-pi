import os, sys, json, base64, io
from pathlib import Path

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import time
import threading

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # plant-detection/
MODEL_PATH       = BASE_DIR / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# ── Model definition (must match training exactly) ────────────────────────────
class PlantDiseaseModel(nn.Module):
    """Custom CNN for plant disease classification."""
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same"),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )
        self.conv_block1     = conv_block(3,   64)
        self.conv_block2     = conv_block(64,  128)
        self.conv_block3     = conv_block(128, 256)
        self.conv_block4     = conv_block(256, 512)
        self.conv_block5     = conv_block(512, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        return self.fc_block(x)

# ── Treatment recommendations per class ───────────────────────────────────────
TREATMENTS = {
    "Pepper__bell___Bacterial_spot": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Remove and destroy infected leaves immediately.",
            "Apply copper-based bactericide (e.g., Bordeaux mixture) every 7–10 days.",
            "Avoid overhead irrigation; use drip watering.",
            "Rotate crops — do not plant peppers in the same spot next season.",
            "Disinfect gardening tools with 10% bleach solution.",
        ],
        "prevention": "Use disease-resistant varieties and certified disease-free seeds.",
    },
    "Pepper__bell___healthy": {
        "status": "healthy",
        "severity": "none",
        "steps": [
            "Your pepper plant looks healthy! 🎉",
            "Water consistently, keeping soil moist but not waterlogged.",
            "Fertilize with a balanced NPK fertilizer every 2–3 weeks.",
            "Monitor for pests and disease weekly.",
            "Ensure at least 6–8 hours of sunlight daily.",
        ],
        "prevention": "Maintain good air circulation and avoid overcrowding.",
    },
    "Potato___Early_blight": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Remove severely infected leaves and dispose of them (do not compost).",
            "Apply a fungicide containing chlorothalonil or mancozeb.",
            "Spray every 7–10 days during humid weather.",
            "Avoid wetting foliage when watering.",
            "Hill up soil around plants to prevent tuber exposure.",
        ],
        "prevention": "Plant certified seed potatoes and practice 3-year crop rotation.",
    },
    "Potato___Late_blight": {
        "status": "diseased",
        "severity": "severe",
        "steps": [
            "⚠️ Act immediately — Late Blight spreads rapidly!",
            "Remove and bag all infected plant material; do not compost.",
            "Apply a systemic fungicide (e.g., metalaxyl or cymoxanil) right away.",
            "Continue spraying every 5–7 days until conditions improve.",
            "Destroy any infected tubers in the soil.",
            "Inform neighboring growers as this disease is highly contagious.",
        ],
        "prevention": "Plant blight-resistant varieties; avoid planting near tomatoes.",
    },
    "Potato___healthy": {
        "status": "healthy",
        "severity": "none",
        "steps": [
            "Your potato plant is healthy! 🎉",
            "Ensure soil stays consistently moist, especially during tuber formation.",
            "Hill soil around stems every 2–3 weeks.",
            "Watch for Colorado potato beetles and aphids.",
            "Avoid over-fertilizing with nitrogen once flowering begins.",
        ],
        "prevention": "Use certified disease-free seed potatoes each season.",
    },
    "Tomato_Bacterial_spot": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Remove affected leaves and stems carefully.",
            "Apply copper-based bactericide spray (copper hydroxide or copper sulfate).",
            "Spray at 7-day intervals, especially after rain.",
            "Avoid working with plants when they are wet.",
            "Disinfect pruning tools between plants.",
        ],
        "prevention": "Use resistant varieties and treat seeds before planting.",
    },
    "Tomato_Early_blight": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Remove lower infected leaves to improve air circulation.",
            "Apply a fungicide (chlorothalonil, mancozeb, or neem oil) weekly.",
            "Mulch around plants to prevent soil splash.",
            "Water at the base, not on leaves.",
            "Stake or cage plants for better airflow.",
        ],
        "prevention": "Rotate tomatoes with non-solanaceous crops every 2–3 years.",
    },
    "Tomato_Late_blight": {
        "status": "diseased",
        "severity": "severe",
        "steps": [
            "⚠️ Emergency: Late Blight can destroy your crop in days!",
            "Remove ALL infected plant tissues immediately and bag them.",
            "Apply a systemic fungicide (metalaxyl, dimethomorph) without delay.",
            "Spray every 5 days during wet, cool weather.",
            "Consider removing the entire plant if infection is widespread.",
        ],
        "prevention": "Plant resistant varieties; avoid evening irrigation.",
    },
    "Tomato_Leaf_Mold": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Improve ventilation — prune excess foliage for airflow.",
            "Reduce humidity in greenhouses below 85%.",
            "Apply fungicide (chlorothalonil or copper-based) every 5–7 days.",
            "Remove and dispose of heavily infected leaves.",
            "Avoid overhead watering.",
        ],
        "prevention": "Grow resistant varieties in well-ventilated conditions.",
    },
    "Tomato_Septoria_leaf_spot": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Remove infected lower leaves first.",
            "Apply fungicide (chlorothalonil or copper fungicide) at 7-day intervals.",
            "Mulch the soil to prevent splashing of fungal spores.",
            "Stake plants to keep foliage off the ground.",
            "Disinfect tools after pruning.",
        ],
        "prevention": "Practice crop rotation and remove plant debris at season end.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Spray plants with a strong jet of water to dislodge mites.",
            "Apply neem oil or insecticidal soap spray every 3–5 days.",
            "Introduce natural predators (lacewings, predatory mites) if available.",
            "Keep plants well-watered — mites thrive in drought-stressed plants.",
            "Remove heavily infested leaves.",
        ],
        "prevention": "Monitor plants regularly; avoid excessive nitrogen fertilization.",
    },
    "Tomato__Target_Spot": {
        "status": "diseased",
        "severity": "moderate",
        "steps": [
            "Remove affected leaves, starting from the bottom of the plant.",
            "Apply fungicide (azoxystrobin or pyraclostrobin) every 7 days.",
            "Ensure good airflow by pruning and staking plants.",
            "Avoid excessive nitrogen fertilizer.",
            "Mulch to reduce soil splashing.",
        ],
        "prevention": "Use resistant varieties and practice crop rotation.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "status": "diseased",
        "severity": "severe",
        "steps": [
            "⚠️ This is a viral disease — there is NO CURE once infected.",
            "Remove and destroy infected plants to prevent spread.",
            "Control whitefly vectors with insecticides (imidacloprid or neem oil).",
            "Use yellow sticky traps to monitor and trap whiteflies.",
            "Install insect-proof nets around plants.",
            "Plant new, resistant varieties after removing infected plants.",
        ],
        "prevention": "Use virus-resistant varieties and control whitefly populations.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "status": "diseased",
        "severity": "severe",
        "steps": [
            "⚠️ Viral disease — no chemical cure is available.",
            "Remove and destroy infected plants immediately.",
            "Wash hands thoroughly after handling infected plants.",
            "Disinfect all tools with 10% bleach or 70% alcohol.",
            "Do not smoke near plants (Tobacco mosaic virus can spread this way).",
            "Aphid control is essential to reduce virus spread.",
        ],
        "prevention": "Use certified virus-free seeds and resistant varieties.",
    },
    "Tomato_healthy": {
        "status": "healthy",
        "severity": "none",
        "steps": [
            "Your tomato plant is healthy! 🎉",
            "Water deeply twice a week, keeping soil evenly moist.",
            "Fertilize with calcium-rich fertilizer to prevent blossom end rot.",
            "Stake or cage plants as they grow.",
            "Pinch out suckers for indeterminate varieties.",
            "Check weekly for early signs of disease or pests.",
        ],
        "prevention": "Rotate crops, use mulch, and maintain consistent watering.",
    },
}

# ── Inference setup ───────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")   # serve on CPU for simplicity
IMAGE_SIZE = (224, 224)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("[BOOT] Loading class names…")
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

print("[BOOT] Loading model weights…")
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES), dropout_rate=0.5)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"[BOOT] Model ready ({len(CLASS_NAMES)} classes)")

# ── OpenCV Camera Setup ───────────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Optimize for lower latency
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1.0)
        self.lock = threading.Lock()
        self.current_frame = None
        
        # Start a background thread to continually grab frames
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            success, image = self.video.read()
            if success:
                with self.lock:
                    self.current_frame = image
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
    
    def get_frame_bytes(self):
        frame = self.get_frame()
        if frame is None:
            return None
        # Compress with slightly lower quality for better streaming performance
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

camera_instance = None
try:
    camera_instance = Camera()
except Exception as e:
    print(f"[WARN] Could not initialize camera: {e}")

def gen_frames():
    global camera_instance
    if camera_instance is None:
        return
    while True:
        frame_bytes = camera_instance.get_frame_bytes()
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Limit to ~30 FPS
        time.sleep(1.0 / 30.0)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run inference
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            top_idx = torch.argmax(probs).item()

        class_name = CLASS_NAMES[top_idx]
        confidence = round(probs[top_idx].item() * 100, 2)

        # Top-3 predictions
        top3 = torch.topk(probs, 3)
        top3_list = [
            {"class": CLASS_NAMES[i], "confidence": round(probs[i].item() * 100, 2)}
            for i in top3.indices.tolist()
        ]

        treatment = TREATMENTS.get(class_name, {
            "status": "unknown",
            "severity": "unknown",
            "steps": ["No specific treatment data available."],
            "prevention": "Consult a local agricultural extension office.",
        })

        # Format display name
        display_name = class_name.replace("___", " — ").replace("__", " — ").replace("_", " ")

        return jsonify({
            "class_name":    class_name,
            "display_name":  display_name,
            "confidence":    confidence,
            "status":        treatment["status"],
            "severity":      treatment["severity"],
            "steps":         treatment["steps"],
            "prevention":    treatment["prevention"],
            "top3":          top3_list,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    global camera_instance
    if camera_instance is None:
        return jsonify({"error": "Camera not available on backend"}), 500
    
    try:
        frame_bytes = camera_instance.get_frame_bytes()
        if frame_bytes is None:
            return jsonify({"error": "Failed to capture frame from camera"}), 500
            
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        
        # Run inference
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            top_idx = torch.argmax(probs).item()

        class_name = CLASS_NAMES[top_idx]
        confidence = round(probs[top_idx].item() * 100, 2)

        # Top-3 predictions
        top3 = torch.topk(probs, 3)
        top3_list = [
            {"class": CLASS_NAMES[i], "confidence": round(probs[i].item() * 100, 2)}
            for i in top3.indices.tolist()
        ]

        treatment = TREATMENTS.get(class_name, {
            "status": "unknown",
            "severity": "unknown",
            "steps": ["No specific treatment data available."],
            "prevention": "Consult a local agricultural extension office.",
        })

        display_name = class_name.replace("___", " — ").replace("__", " — ").replace("_", " ")

        return jsonify({
            "class_name":    class_name,
            "display_name":  display_name,
            "confidence":    confidence,
            "status":        treatment["status"],
            "severity":      treatment["severity"],
            "steps":         treatment["steps"],
            "prevention":    treatment["prevention"],
            "top3":          top3_list,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    global camera_instance
    return jsonify({
        "status": "ok", 
        "classes": len(CLASS_NAMES),
        "camera_available": camera_instance is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
