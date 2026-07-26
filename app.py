import os
import colorsys
from flask import Flask, request, jsonify, render_template
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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


def classify_film(r, g, b):
    """
    Core smart-film logic:
      Dark / slight purple  -> Fresh
      Light purple          -> Moderately Spoiled
      White                 -> Spoiled
    Classification is done using HSV (hue, saturation, value) which is far
    more stable than raw RGB thresholds under different lighting.
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue_deg = h * 360

    # Purple / violet hue band (roughly 240 - 320 degrees)
    is_purple_hue = 230 <= hue_deg <= 330

    # White film: very low saturation + very high brightness, regardless of hue
    if s <= 0.18 and v >= 0.78:
        status = "Spoiled"
        confidence = round(min(99, (v * 100) - (s * 40)), 1)

    # Light purple: purple hue, moderate saturation, high-ish brightness
    elif is_purple_hue and s <= 0.45 and v >= 0.55:
        status = "Moderately Spoiled"
        confidence = round(min(95, 60 + (s * 50)), 1)

    # Dark / deep purple: purple hue, strong saturation, lower brightness
    elif is_purple_hue and s > 0.35:
        status = "Fresh"
        confidence = round(min(98, 65 + (s * 40) - (v * 10)), 1)

    else:
        # Fallback: use brightness alone if hue is ambiguous (e.g. shadows)
        if v >= 0.75:
            status = "Spoiled"
            confidence = 55.0
        elif v >= 0.5:
            status = "Moderately Spoiled"
            confidence = 55.0
        else:
            status = "Fresh"
            confidence = 55.0

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

    r, g, b = get_average_color(filepath)
    status, confidence, hue, sat, val = classify_film(r, g, b)
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