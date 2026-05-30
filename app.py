from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

# ==========================
# Upload Folder
# ==========================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


# ==========================
# Film Analysis
# ==========================

def analyze_film(image_path):

    img = cv2.imread(image_path)

    if img is None:

        return {

            "film_color": "Unknown",
            "status": "Error",
            "freshness": "0%",
            "level": "Unknown",
            "ph": "-",
            "remaining_days": "-",
            "precaution": "Image not readable",
            "storage": "-"

        }

    img = cv2.resize(
        img,
        (400, 400)
    )

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # Purple Film Range

    lower_purple = np.array(
        [110, 40, 40]
    )

    upper_purple = np.array(
        [170, 255, 255]
    )

    # White Film Range

    lower_white = np.array(
        [0, 0, 160]
    )

    upper_white = np.array(
        [180, 70, 255]
    )

    purple_mask = cv2.inRange(
        hsv,
        lower_purple,
        upper_purple
    )

    white_mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white
    )

    purple_pixels = cv2.countNonZero(
        purple_mask
    )

    white_pixels = cv2.countNonZero(
        white_mask
    )

    total_pixels = (
        img.shape[0]
        * img.shape[1]
    )

    purple_percent = (
        purple_pixels / total_pixels
    ) * 100

    white_percent = (
        white_pixels / total_pixels
    ) * 100

    print(
        "Purple %",
        purple_percent
    )

    print(
        "White %",
        white_percent
    )

    # ======================
    # Fresh
    # ======================

    if purple_percent > white_percent:

        return {

            "film_color":
            "Purple",

            "status":
            "Fresh",

            "freshness":
            "95%",

            "level":
            "Low",

            "ph":
            "6.7",

            "remaining_days":
            "5 Days",

            "precaution":
            "Safe to Consume",

            "storage":
            "Store below 4°C"

        }

    # ======================
    # Spoiled
    # ======================

    else:

        return {

            "film_color":
            "White",

            "status":
            "Spoiled",

            "freshness":
            "20%",

            "level":
            "High",

            "ph":
            "4.5",

            "remaining_days":
            "0 Days",

            "precaution":
            "Do Not Consume",

            "storage":
            "Discard Immediately"

        }


# ==========================
# Home Page
# ==========================

@app.route("/")

def home():

    return render_template(
        "index.html"
    )


# ==========================
# Upload Image
# ==========================

@app.route(
    "/upload",
    methods=["POST"]
)

def upload():

    file = request.files.get(
        "image"
    )

    if not file:

        return render_template(
            "index.html"
        )

    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        file.filename

    )

    file.save(filepath)

    result = analyze_film(
        filepath
    )

    return render_template(

        "index.html",

        result=result,

        image_path=filepath

    )


# ==========================
# Live Camera Scan
# ==========================

@app.route(
    "/scan",
    methods=["POST"]
)

def scan():

    data = request.json

    image_data = data["image"]

    image_data = image_data.split(
        ","
    )[1]

    image_bytes = base64.b64decode(
        image_data
    )

    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        "captured.png"

    )

    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            image_bytes
        )

    result = analyze_film(
        filepath
    )

    return jsonify(
        result
    )


# ==========================
# Run App
# ==========================

if __name__ == "__main__":

    app.run(
        debug=True
    )