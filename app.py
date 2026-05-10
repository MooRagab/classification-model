from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import load_class_names, load_model, predict_top_k

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

def _inject_styles(mode: str):
    if mode == "Dark":
        bg = """
        radial-gradient(circle at 8% 8%, rgba(236, 72, 153, 0.18), transparent 28%),
        radial-gradient(circle at 92% 0%, rgba(56, 189, 248, 0.25), transparent 24%),
        linear-gradient(140deg, #020617 0%, #030712 50%, #0a0f1d 100%)
        """
        hero_bg = "linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.75))"
        hero_title = "#f8fafc"
        hero_sub = "#cbd5e1"
        card_bg = "linear-gradient(120deg, rgba(17, 24, 39, 0.92), rgba(31, 41, 55, 0.88))"
        card_border = "rgba(34, 211, 238, 0.35)"
        pred_label = "#e0f2fe"
        pred_score = "#22d3ee"
        panel_bg = "rgba(15, 23, 42, 0.72)"
        panel_border = "rgba(34, 211, 238, 0.30)"
        section_title = "#e2e8f0"
        accent = "#22d3ee"
        chip_bg = "rgba(34, 211, 238, 0.14)"
        chip_fg = "#67e8f9"
    else:
        bg = """
        radial-gradient(circle at 12% 12%, rgba(248, 113, 113, 0.14), transparent 28%),
        radial-gradient(circle at 88% 2%, rgba(14, 165, 233, 0.14), transparent 26%),
        linear-gradient(145deg, #fef9f3 0%, #fff7ed 40%, #f8fafc 100%)
        """
        hero_bg = "linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(255, 247, 237, 0.92))"
        hero_title = "#111827"
        hero_sub = "#475569"
        card_bg = "linear-gradient(120deg, rgba(255, 255, 255, 0.98), rgba(250, 250, 249, 0.94))"
        card_border = "rgba(251, 146, 60, 0.35)"
        pred_label = "#111827"
        pred_score = "#ea580c"
        panel_bg = "rgba(255, 255, 255, 0.82)"
        panel_border = "rgba(251, 146, 60, 0.24)"
        section_title = "#111827"
        accent = "#ea580c"
        chip_bg = "rgba(251, 146, 60, 0.12)"
        chip_fg = "#c2410c"

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

        .block-container {{
            max-width: 980px;
            padding-top: 4.2rem;
            padding-bottom: 2.5rem;
        }}

        .hero-card {{
            border-radius: 24px;
            padding: 1.2rem 1.35rem 1.15rem 1.35rem;
            background: {hero_bg};
            border: 1px solid {panel_border};
            box-shadow: 0 20px 40px rgba(2, 6, 23, 0.12);
            backdrop-filter: blur(8px);
            margin-bottom: 1.05rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-card:before {{
            content: "";
            position: absolute;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: {chip_bg};
            right: -35px;
            top: -35px;
        }}

        .hero-title {{
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
            color: {hero_title};
        }}

        .hero-subtitle {{
            font-size: 0.98rem;
            color: {hero_sub};
            margin: 0 0 0.75rem 0;
        }}
        .hero-row {{
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }}
        .hero-chip {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background: {chip_bg};
            color: {chip_fg};
            border: 1px solid {panel_border};
        }}

        .soft-panel {{
            border-radius: 18px;
            padding: 0.9rem 1rem 0.3rem 1rem;
            background: {panel_bg};
            border: 1px solid {panel_border};
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(8px);
        }}

        .section-title {{
            font-size: 1.15rem;
            font-weight: 750;
            color: {section_title};
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.01em;
            text-transform: uppercase;
            font-size: 0.83rem;
        }}

        .pred-card {{
            border-radius: 12px;
            padding: 0.8rem 0.92rem;
            background: {card_bg};
            border: 1px solid {card_border};
            margin-bottom: 0.55rem;
            border-left: 4px solid {accent};
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

        [data-testid="stFileUploader"] {{
            background: {panel_bg};
            border: 1px solid {panel_border};
            border-radius: 14px;
            padding: 0.35rem 0.45rem 0.15rem 0.45rem;
        }}

        [data-testid="stSidebar"] {{
            border-right: 1px solid {panel_border};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {accent}, #fb7185) !important;
        }}
        </style>
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

st.markdown(
    f"""
    <div class="hero-card">
      <div class="hero-title">Scene Vision Console</div>
      <p class="hero-subtitle">Smart scene classification powered by transfer learning. Upload an image and get ranked predictions instantly.</p>
      <div class="hero-row">
        <span class="hero-chip">Backbone: MobileNetV2</span>
        <span class="hero-chip">Classes: {len(class_names)}</span>
        <span class="hero-chip">Top-3 Confidence</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

    st.markdown('<div class="soft-panel"><p class="section-title">Top 3 Predictions</p>', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

    chart_df = pd.DataFrame(
        {"Class": [label for label, _ in results], "Confidence": [score * 100 for _, score in results]}
    ).set_index("Class")
    st.markdown('<div class="soft-panel"><p class="section-title">Confidence Chart</p>', unsafe_allow_html=True)
    st.bar_chart(chart_df, color="#14b8a6")
    st.markdown("</div>", unsafe_allow_html=True)
