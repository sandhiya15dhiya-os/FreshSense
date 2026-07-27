import os
import colorsys
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image
from werkzeug.utils import secure_filename

from features import extract_features

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pre-load the face detector once at startup (not per-request) for speed
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load the trained ML model if it exists (run train_model.py to create it).
# If it's missing, the app falls back to the hardcoded HSV rules below so
# it still works out of the box.
MODEL_PATH = "model.pkl"
ML_MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def contains_face(image_path):
    """
    Returns True if a human face is detected in the image. Used to block
    people from accidentally (or intentionally) scanning a selfie/face
    instead of an actual food/film sample.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Faces smaller than this are unlikely to be the main subject
    min_size = max(40, int(min(h, w) * 0.12))
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size)
    )
    return len(faces) > 0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_average_color(image_path, crop_ratio=0.5):
    """
    Crop the center region of the image (where the smart film patch is
    expected to be) and return its average RGB color.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # Crop center region so background clutter does not skew the reading
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))

    # Downscale for fast, stable average
    small = cropped.resize((50, 50))
    pixels = list(small.getdata())
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    return r, g, b


def classify_with_model(image_path, r, g, b):
    """
    Uses the trained Random Forest model if available (much more accurate,
    learns from real photos). Falls back to the hardcoded HSV rules if
    model.pkl hasn't been trained yet (see train_model.py).
    Returns: status, confidence, hue, sat, val
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue_deg = round(h * 360, 1)

    if ML_MODEL is not None:
        feats = extract_features(image_path).reshape(1, -1)
        probs = ML_MODEL.predict_proba(feats)[0]
        classes = ML_MODEL.classes_
        best_idx = int(np.argmax(probs))
        status = classes[best_idx]
        confidence = round(float(probs[best_idx]) * 100, 1)
        return status, confidence, hue_deg, round(s, 2), round(v, 2)

    # ---- Fallback: hardcoded HSV rules (used only if model.pkl is missing) ----
    return classify_film(r, g, b)


def classify_film(r, g, b):
    """
    Core smart-film logic:
      Purple (dark or light)   -> Fresh
      Yellow / pale yellow     -> Moderately Spoiled
      White                    -> Spoiled
    Classification is done using HSV (hue, saturation, value) which is far
    more stable than raw RGB thresholds under different lighting.
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue_deg = h * 360

    is_white = s <= 0.12 and v >= 0.85
    is_purple = 230 <= hue_deg <= 330 and s >= 0.15
    is_yellow = 25 <= hue_deg <= 75

    if is_white:
        # Achromatic (near colorless) + bright = white film
        status = "Spoiled"
        confidence = round(min(99, (v * 100) - (s * 30)), 1)

    elif is_purple:
        # Any purple tint, dark or light, counts as Fresh
        status = "Fresh"
        confidence = round(min(98, 55 + (s * 50)), 1)

    elif is_yellow:
        # Pale yellow / cream / straw-yellow tint = moderately spoiled
        # Confidence is higher the closer the hue sits to pure yellow (45deg)
        closeness = max(0.0, 1 - abs(hue_deg - 45) / 30)
        confidence = round(min(95, 55 + (closeness * 30) + (s * 20)), 1)
        status = "Moderately Spoiled"

    else:
        # Ambiguous color (greenish/brownish/etc.) - fall back on brightness
        if v >= 0.8 and s <= 0.25:
            status = "Spoiled"
        elif s <= 0.3:
            status = "Moderately Spoiled"
        else:
            status = "Fresh"
        confidence = 50.0

    return status, max(50.0, min(confidence, 99.0)), round(hue_deg, 1), round(s, 2), round(v, 2)


def predict_ph(status):
    ranges = {
        "Fresh": (6.6, 6.8),
        "Moderately Spoiled": (6.0, 6.4),
        "Spoiled": (4.4, 5.5),
    }
    low, high = ranges[status]
    return round((low + high) / 2, 2), low, high


def predict_shelf_life(status):
    shelf = {
        "Fresh": "3 - 4 days (store below 4°C)",
        "Moderately Spoiled": "Consume within 12 - 24 hours",
        "Spoiled": "0 days - Do not consume",
    }
    return shelf[status]


def get_precautions(status):
    precautions = {
        "Fresh": [
            "Store immediately in a refrigerator below 4°C.",
            "Keep the container tightly sealed to avoid contamination.",
            "Avoid leaving it out at room temperature for long.",
            "Re-check freshness if stored for more than 3 days.",
        ],
        "Moderately Spoiled": [
            "Consume as soon as possible, preferably within a few hours.",
            "Boil milk thoroughly before use to reduce bacterial load.",
            "Do not give to infants, elderly, or immunocompromised people.",
            "Avoid mixing with fresh stock to prevent cross-spoilage.",
        ],
        "Spoiled": [
            "Do NOT consume - discard the product immediately.",
            "Do not use even after boiling, toxins may already be present.",
            "Wash the storage container thoroughly before reuse.",
            "Check other items stored nearby for possible spoilage spread.",
        ],
    }
    return precautions[status]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Block selfies / face photos - this scanner is for food/film samples only
    if contains_face(filepath):
        os.remove(filepath)
        return jsonify({
            'error': "Face detected. Please scan a food item or smart film, not a face/selfie."
        }), 400

    r, g, b = get_average_color(filepath)
    status, confidence, hue, sat, val = classify_with_model(filepath, r, g, b)
    ph_avg, ph_low, ph_high = predict_ph(status)
    shelf_life = predict_shelf_life(status)
    precautions = get_precautions(status)

    return jsonify({
        'status': status,
        'confidence': confidence,
        'ph_prediction': {
            'average': ph_avg,
            'range': f"{ph_low} - {ph_high}"
        },
        'shelf_life': shelf_life,
        'precautions': precautions,
        'debug': {
            'avg_rgb': [round(r, 1), round(g, 1), round(b, 1)],
            'hue': hue,
            'saturation': sat,
            'value': val
        },
        'image_url': '/' + filepath.replace('\\', '/')
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)