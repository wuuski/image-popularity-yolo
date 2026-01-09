import io
import pickle

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import lightgbm as lgb

# --------- CONFIG ---------
YOLO_WEIGHTS = "yolo11n.pt"          # YOLO model weights
LGBM_MODEL_PATH = "lgbm_model.txt"   # trained LightGBM Booster
FEATURE_COLS_PATH = "feature_cols.pkl"  # list of feature column names

st.set_page_config(
    page_title="Popularity Prediction (YOLO + LightGBM)",
    layout="wide",
    page_icon="📸",
)
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Tighten up default margins / fonts a bit
st.markdown(
    """
    <style>
        .main {
            padding-top: 1.2rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
        }
        section[data-testid="stSidebar"] {
            width: 260px !important;
        }
        h1, h2, h3 {
            margin-bottom: 0.4rem;
        }
        .small-label {
            font-size: 0.9rem;
            color: #999999;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- CACHED LOADERS ---------


@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_WEIGHTS)


@st.cache_resource
def load_lgbm_and_features():
    # LightGBM booster
    booster = lgb.Booster(model_file=LGBM_MODEL_PATH)

    # feature column list (order MUST match training)
    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    return booster, feature_cols


# --------- FEATURE UTILITIES ---------


def compute_color_features(img_np: np.ndarray) -> dict:
    """
    Compute simple color features + a 'colorfulness' metric.
    Assumes img_np is RGB uint8.
    """
    img = img_np.astype(np.float32)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    mean_R = float(R.mean())
    mean_G = float(G.mean())
    mean_B = float(B.mean())

    std_R = float(R.std())
    std_G = float(G.std())
    std_B = float(B.std())

    # Hasler & Suesstrunk-style colorfulness
    rg = R - G
    yb = 0.5 * (R + G) - B

    std_rg = rg.std()
    std_yb = yb.std()
    mean_rg = np.abs(rg.mean())
    mean_yb = np.abs(yb.mean())

    colorfulness = float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    return {
        "mean_R": mean_R,
        "mean_G": mean_G,
        "mean_B": mean_B,
        "std_R": std_R,
        "std_G": std_G,
        "std_B": std_B,
        "colorfulness": colorfulness,
    }


def process_pil_image(pil_img: Image.Image, yolo_model, feature_cols):
    """
    Run YOLO on a PIL image, build the feature vector (matching training),
    and draw bounding boxes + a 3×3 grid on the image.

    Returns:
      X_input : 1-row DataFrame with columns = feature_cols
      vis_rgb : annotated RGB numpy array (H,W,3)
      feat_dict : the raw feature dict (for display later)
    """
    # Base feature dict
    feat_dict = {col: 0.0 for col in feature_cols}

    # Colors
    img_rgb = pil_img.convert("RGB")
    img_np = np.array(img_rgb)
    h, w = img_np.shape[:2]
    cell_w = w / 3.0
    cell_h = h / 3.0

    # Insert color features
    color_feats = compute_color_features(img_np)
    for k, v in color_feats.items():
        if k in feat_dict:
            feat_dict[k] = v

    # YOLO
    results = yolo_model(img_rgb)[0]
    boxes = results.boxes
    names = results.names

    # For drawing: work in BGR for cv2
    vis = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)

    if boxes is not None and len(boxes) > 0:
        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            class_name = names[int(cls_id)]

            # Global count feature
            global_key = f"count_{class_name}"
            if global_key in feat_dict:
                feat_dict[global_key] += 1

            # Bounding box and center
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Determine grid row/col (0,1,2)
            col = min(int(cx // cell_w), 2)
            row = min(int(cy // cell_h), 2)

            pos_key = f"count_{class_name}_r{row}_c{col}"
            if pos_key in feat_dict:
                feat_dict[pos_key] += 1

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label text
            label = f"{class_name} (r{row},c{col})"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis,
                (x1, max(0, y1 - th - baseline)),
                (x1 + tw, y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                vis,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # Draw 3×3 grid
    for i in range(1, 3):
        x = int(i * cell_w)
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)

        y = int(i * cell_h)
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # Build DataFrame with correct column order
    X_input = pd.DataFrame([[feat_dict[c] for c in feature_cols]], columns=feature_cols)

    return X_input, vis_rgb, feat_dict


# --------- MAIN UI ---------

st.title("📸 Popularity Prediction (YOLO + LightGBM)")

st.markdown(
    "Upload an image. The app runs YOLO to detect objects, aggregates counts in a 3×3 "
    "grid, adds simple color features, and feeds them into a trained LightGBM model "
    "to predict whether the image is **popular** or **not popular**."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

# Load models
model_load_error = None
try:
    yolo_model = load_yolo_model()
    lgbm_model, feature_cols = load_lgbm_and_features()
except Exception as e:
    yolo_model = None
    lgbm_model = None
    feature_cols = None
    model_load_error = str(e)

if model_load_error is not None:
    st.error(
        "Error loading models. Make sure the following files exist in the same folder as `app.py`:\n"
        f"- `{YOLO_WEIGHTS}`\n"
        f"- `{LGBM_MODEL_PATH}`\n"
        f"- `{FEATURE_COLS_PATH}`\n\n"
        f"Details: {model_load_error}"
    )

if uploaded_file is not None and yolo_model is not None and lgbm_model is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running YOLO + LightGBM on your image..."):
        X_input, vis_rgb, feat_dict = process_pil_image(
            pil_img, yolo_model, feature_cols
        )

        # LightGBM prediction: returns probability for class 1 (popular)
        prob_pop = float(lgbm_model.predict(X_input)[0])
        prob_not = 1.0 - prob_pop
        popularity_score = prob_pop * 100.0

    # Layout: prediction (left) and YOLO visualization (right)
    left_col, right_col = st.columns([0.95, 1.05])

    with left_col:
        st.subheader("Prediction")

        label = "✨ Popular" if prob_pop >= 0.5 else "😐 Not Popular"

        st.markdown(f"### {label}")

        # Big score
        st.markdown(
            f"""
            <div class="small-label">Popularity score (0–100)</div>
            <div style="font-size: 46px; font-weight: 700; margin-top: -0.1rem; margin-bottom: 0.4rem;">
                {popularity_score:0.1f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Model confidence for Popular (1)**")

        # Probability "progress bar"
        st.progress(min(max(prob_pop, 0.0), 1.0))

        st.write(
            f"**Popular (1):** `{prob_pop:0.3f}` &nbsp;&nbsp;&nbsp; "
            f"**Not Popular (0):** `{prob_not:0.3f}`"
        )

        # Feature vector (non-zero first)
        feat_series = pd.Series(feat_dict, name="value")
        feat_series.index.name = "feature"
        df_feats = feat_series.to_frame()

        non_zero = df_feats[df_feats["value"] != 0]
        zero = df_feats[df_feats["value"] == 0]
        df_display = pd.concat([non_zero, zero])

        with st.expander("Show raw feature vector used for this prediction"):
            st.caption("Non-zero features are listed first.")
            st.dataframe(
                df_display,
                use_container_width=True,
            )

    with right_col:
        st.subheader("YOLO detections with 3×3 grid")
        st.image(vis_rgb, caption="Detections + 3×3 grid", use_container_width=True)

elif uploaded_file is not None and model_load_error is None:
    st.warning("Models are still loading or unavailable.")

# -------------------------
# Project Write-Up Section
# -------------------------
import streamlit as st

with st.expander("📘 Project Write-Up / Reflection", expanded=False):
    st.markdown(
        """
### Project Reflection

This project ended up being a lot more involved than I expected when I first decided to build a “popularity predictor” from images. The goal was simple: take aesthetic and compositional cues from a photo and estimate whether it would be considered “popular.” But actually getting the model to understand anything meaningful required a long chain of steps—starting from object detection, to feature engineering, to training and evaluating a LightGBM classifier.

I began by running each image through YOLO to extract structured information. YOLO doesn’t just tell you which objects are present; it also gives bounding boxes and coordinates. To impose some notion of spatial layout, I divided each image into a 3×3 grid and counted how many times each object type appeared in each cell. This alone produced more than 700 features.

I also added simple color statistics: the mean and standard deviation of the RGB channels and a “colorfulness” metric. After merging these features with the label dataset and aligning filenames, I trained a LightGBM model with subsampled data to avoid long training times.

Most of the iteration was spent debugging feature-alignment issues, YOLO inconsistencies, and unintentionally exploding the feature dimension (722 features is… a lot). A surprising amount of time also went into building the visualization for the web app and adjusting margins so the interface actually looked normal on screen.

### Most Important Features

Even though the model isn’t perfect, LightGBM gives a helpful ranking of what it thinks actually matters. Several patterns emerged:

##### 1. Object Presence, Especially People

The model consistently relies on where people appear in the image. Grids where the main person is centered or in specific cells tended to correlate with higher predicted popularity. This lines up with intuition — photos with a clear subject are typically more engaging.

##### 2. Counts of “car,” “bus,” and other large objects

Certain object types had outsized influence. In some datasets, photos with vehicles or street settings ended up associated with high popularity, probably because these appear frequently in well-liked urban or night-time photos.

##### 3. Colorfulness and Brightness

One of the strongest non-position features was colorfulness. Images with balanced or mid-range colorfulness tended to be predicted as more popular. Extremely dark or washed-out images skewed downward. This helps explain why many of your newer test images received lower scores — the model has implicitly learned that darker, low-contrast nighttime photos correlate with low popularity.

#### Conclusion

Overall, the project ended up revealing more about the dataset than about aesthetics themselves. The model learned strong correlations with lighting, subject placement, and color balance — all factors that genuinely influence how “popular” an image feels. At the same time, its uneven performance on darker nighttime photos shows how dependent it is on the biases in the training set.


        """,
        unsafe_allow_html=True,
    )