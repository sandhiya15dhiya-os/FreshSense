"""
train_model.py
----------------
Trains a Random Forest classifier for FreshSense AI's smart-film analysis.

HOW TO USE:
1. Put your OWN real photos of the film patch into these folders:
     dataset/fresh/       -> purple film photos (dark or light purple)
     dataset/moderate/    -> yellow / pale-yellow film photos
     dataset/spoiled/     -> white film photos
   Aim for at least 20-30 photos per class, taken under different lighting,
   angles, and camera distances for best real-world accuracy.

2. Run:
     python train_model.py

3. This creates model.pkl in the project root. app.py automatically loads
   it and uses the trained model instead of the old hardcoded HSV rules.

NOTE: If you have few or no real photos yet, this script still works -
it bootstraps a synthetic dataset so the app has a working model from day
one. But real photos of YOUR actual smart film will make it far more
accurate than synthetic colors alone, so add them before your final demo.
"""

import os
import glob
import numpy as np
import colorsys
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from features import extract_features

DATASET_DIR = "dataset"
CLASSES = {
    "fresh": "Fresh",
    "moderate": "Moderately Spoiled",
    "spoiled": "Spoiled",
}
IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

SYNTHETIC_SAMPLES_PER_CLASS = 150
SYNTHETIC_DIR = "_synthetic_bootstrap"


def load_real_images():
    """Load any real photos the user has placed in dataset/<class>/."""
    X, y = [], []
    counts = {}
    for folder, label in CLASSES.items():
        folder_path = os.path.join(DATASET_DIR, folder)
        files = []
        for ext in IMG_EXTENSIONS:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        counts[label] = len(files)
        for f in files:
            try:
                X.append(extract_features(f))
                y.append(label)
            except Exception as e:
                print(f"  [skip] Could not read {f}: {e}")
    return X, y, counts


def _hsv_patch_to_image(h, s, v, size=64, noise=0.04):
    """Create a small solid-color patch (with slight pixel noise) from an
    HSV color, mimicking a photographed film sample."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    base = np.array([r, g, b], dtype=np.float32)
    img = np.tile(base, (size, size, 1))
    img += np.random.normal(0, noise, img.shape)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def generate_synthetic_dataset():
    """
    Generates a bootstrap dataset of synthetic film-color patches so the
    model has *something* to learn from even before real photos exist.
    Replace/supplement with real photos in dataset/ for best accuracy.
    """
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    X, y = [], []

    ranges = {
        "Fresh":               {"hue": (235, 325), "sat": (0.15, 0.9), "val": (0.12, 0.85)},
        "Moderately Spoiled":  {"hue": (25, 75),    "sat": (0.05, 0.8), "val": (0.55, 0.98)},
        "Spoiled":             {"hue": (0, 360),    "sat": (0.0, 0.12), "val": (0.82, 1.0)},
    }

    idx = 0
    for label, rng in ranges.items():
        for i in range(SYNTHETIC_SAMPLES_PER_CLASS):
            hue = np.random.uniform(*rng["hue"]) / 360.0
            sat = np.random.uniform(*rng["sat"])
            val = np.random.uniform(*rng["val"])
            img = _hsv_patch_to_image(hue % 1.0, sat, val)
            path = os.path.join(SYNTHETIC_DIR, f"{label.replace(' ', '_')}_{idx}.png")
            img.save(path)
            X.append(extract_features(path))
            y.append(label)
            idx += 1

    return X, y


def main():
    print("Loading real photos from dataset/ ...")
    X_real, y_real, counts = load_real_images()
    for label, n in counts.items():
        print(f"  {label:22s}: {n} real photo(s)")

    print("\nGenerating synthetic bootstrap samples ...")
    X_syn, y_syn = generate_synthetic_dataset()
    print(f"  Generated {len(X_syn)} synthetic samples "
          f"({SYNTHETIC_SAMPLES_PER_CLASS} per class)")

    X = X_real + X_syn
    y = y_real + y_syn

    if sum(counts.values()) == 0:
        print("\n⚠️  No real photos found in dataset/fresh, dataset/moderate, "
              "dataset/spoiled.")
        print("   Training on synthetic colors only for now. Add real photos "
              "and re-run this script before your final demo for best accuracy.\n")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier ...")
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "model.pkl")
    print("Saved trained model to model.pkl")


if __name__ == "__main__":
    main()