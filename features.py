import colorsys
import numpy as np
from PIL import Image


def _center_crop(img, crop_ratio=0.5):
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def extract_features(image_path):
    """
    Extract a feature vector describing the color of the film patch in the
    center of the image. Combines simple average color stats with a hue
    histogram so the model can learn shape of the color distribution, not
    just a single average pixel (much more robust to shadows/highlights).

    Returns a 1D numpy array of features.
    """
    img = Image.open(image_path).convert('RGB')
    cropped = _center_crop(img, crop_ratio=0.5)
    small = cropped.resize((64, 64))
    arr = np.asarray(small).astype(np.float32) / 255.0  # (64,64,3)

    r = arr[:, :, 0].flatten()
    g = arr[:, :, 1].flatten()
    b = arr[:, :, 2].flatten()

    # Convert every pixel to HSV
    hsv = np.array([colorsys.rgb_to_hsv(rr, gg, bb) for rr, gg, bb in zip(r, g, b)])
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # Basic stats
    features = [
        r.mean(), g.mean(), b.mean(),
        r.std(), g.std(), b.std(),
        h.mean(), s.mean(), v.mean(),
        h.std(), s.std(), v.std(),
    ]

    # 8-bin hue histogram (captures whether the patch is purple/yellow/white
    # dominant even if lighting varies) + saturation & value histograms
    hue_hist, _ = np.histogram(h, bins=8, range=(0, 1), density=True)
    sat_hist, _ = np.histogram(s, bins=6, range=(0, 1), density=True)
    val_hist, _ = np.histogram(v, bins=6, range=(0, 1), density=True)

    features.extend(hue_hist.tolist())
    features.extend(sat_hist.tolist())
    features.extend(val_hist.tolist())

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = (
    ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std',
     'h_mean', 's_mean', 'v_mean', 'h_std', 's_std', 'v_std']
    + [f'hue_bin_{i}' for i in range(8)]
    + [f'sat_bin_{i}' for i in range(6)]
    + [f'val_bin_{i}' for i in range(6)]
)