from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import load_class_names, load_model, predict_top_k

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

def _inject_styles(mode: str):
    if mode == "Dark":
        bg = """
        radial-gradient(circle at 10% 10%, rgba(14, 165, 233, 0.22), transparent 35%),
        radial-gradient(circle at 90% 0%, rgba(251, 191, 36, 0.20), transparent 30%),
        linear-gradient(160deg, #020617 0%, #0f172a 55%, #111827 100%)
        """
        hero_bg = "rgba(15, 23, 42, 0.72)"
        hero_title = "#e2e8f0"
        hero_sub = "#cbd5e1"
        card_bg = "rgba(30, 41, 59, 0.9)"
        card_border = "rgba(148, 163, 184, 0.22)"
        pred_label = "#e2e8f0"
        pred_score = "#2dd4bf"
    else:
        bg = """
        radial-gradient(circle at 10% 10%, rgba(255, 214, 102, 0.22), transparent 35%),
        radial-gradient(circle at 90% 0%, rgba(38, 198, 218, 0.18), transparent 30%),
        linear-gradient(160deg, #f8fafc 0%, #eef2ff 52%, #ecfeff 100%)
        """
        hero_bg = "rgba(255, 255, 255, 0.72)"
        hero_title = "#0f172a"
        hero_sub = "#334155"
        card_bg = "rgba(255,255,255,0.85)"
        card_border = "rgba(148, 163, 184, 0.2)"
        pred_label = "#0f172a"
        pred_score = "#0f766e"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;800&display=swap');

        html, body, [class*="css"]  {{
            font-family: 'Outfit', sans-serif;
        }}

        .stApp {{
            background: {bg};
        }}

        .hero-card {{
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            background: {hero_bg};
            border: 1px solid rgba(255,255,255,0.32);
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.10);
            backdrop-filter: blur(8px);
            margin-bottom: 0.8rem;
        }}

        .hero-title {{
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
            letter-spacing: -0.01em;
            color: {hero_title};
        }}

        .hero-subtitle {{
            font-size: 1rem;
            color: {hero_sub};
            margin: 0;
        }}

        .pred-card {{
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            background: {card_bg};
            border: 1px solid {card_border};
            margin-bottom: 0.65rem;
        }}

        .pred-label {{
            font-weight: 700;
            color: {pred_label};
            font-size: 1.02rem;
        }}

        .pred-score {{
            font-weight: 700;
            color: {pred_score};
            font-size: 0.95rem;
            float: right;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">Image Classification with MobileNetV2</div>
      <p class="hero-subtitle">Upload an image and get top 3 predictions with confidence scores.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = Path("artifacts/model/best_model.keras")
class_names_path = Path("artifacts/model/class_names.json")

if not model_path.exists() or not class_names_path.exists():
    st.error(
        "Model artifacts not found. Train the model first:\n"
        "`python -m src.train --data_dir data/raw --epochs 20 --batch_size 32`"
    )
    st.stop()

@st.cache_resource
def _load_assets():
    model = load_model(str(model_path))
    class_names = load_class_names(str(class_names_path))
    return model, class_names


model, class_names = _load_assets()


def _get_sample_image_path() -> Path | None:
    roots = [Path("data/raw/test"), Path("data/raw/train")]
    for root in roots:
        if not root.exists():
            continue
        for cls in sorted([p for p in root.iterdir() if p.is_dir()]):
            images = list(cls.glob("*.jpg")) + list(cls.glob("*.jpeg")) + list(cls.glob("*.png")) + list(cls.glob("*.webp"))
            if images:
                return images[0]
    return None


with st.sidebar:
    theme_mode = st.selectbox("Theme", ["Light", "Dark"], index=0)
    st.markdown("### Model Info")
    st.write("Backbone: `MobileNetV2`")
    st.write(f"Classes: **{len(class_names)}**")
    st.write("Input size: `224 x 224`")
    st.caption("Tip: Best results come from outdoor scene images.")
    sample_clicked = st.button("Try Sample Image")

_inject_styles(theme_mode)

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
image = None
image_caption = "Uploaded Image"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif sample_clicked:
    sample_image_path = _get_sample_image_path()
    if sample_image_path is not None:
        image = Image.open(sample_image_path).convert("RGB")
        image_caption = f"Sample Image: {sample_image_path.name}"
    else:
        st.warning("No sample image found under data/raw/train or data/raw/test.")

if image is not None:
    st.image(image, caption=image_caption, width="stretch")

    with st.spinner("Predicting..."):
        results = predict_top_k(model, image, class_names, top_k=3, image_size=(224, 224))

    st.subheader("Top 3 Predictions")
    for idx, (label, score) in enumerate(results, start=1):
        st.markdown(
            f"""
            <div class="pred-card">
              <span class="pred-label">{idx}. {label}</span>
              <span class="pred-score">{score * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(score), text=f"Confidence: {score * 100:.2f}%")

    chart_df = pd.DataFrame(
        {"Class": [label for label, _ in results], "Confidence": [score * 100 for _, score in results]}
    ).set_index("Class")
    st.markdown("### Confidence Chart")
    st.bar_chart(chart_df, color="#14b8a6")
