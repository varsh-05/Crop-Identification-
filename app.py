"""
CropID — Multimodal Crop Identification & Nutrition System
Run:  python -m streamlit run app.py
"""

import streamlit as st
import os, base64, warnings, hashlib, ssl
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance
from groq import Groq
from dotenv import load_dotenv
import requests
from io import BytesIO
import urllib.request

warnings.filterwarnings('ignore')

# Fix SSL issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="CropID — AI Crop System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d2b1a 0%, #163d28 100%);
}
[data-testid="stSidebar"] * { color: #d4edda !important; }
.stApp { background: #f4f8f5; }

/* ── Compact metric cards ── */
.metric-row { display:flex; gap:10px; margin:12px 0; }
.metric-card {
    flex:1; border-radius:10px; padding:10px 12px;
    text-align:center; color:white;
}
.metric-card .m-val {
    font-family:'Syne',sans-serif; font-size:1.05rem;
    font-weight:800; line-height:1.3; word-break:break-word;
}
.metric-card .m-lbl {
    font-size:0.65rem; opacity:0.85;
    letter-spacing:.06em; text-transform:uppercase; margin-top:3px;
}

/* ── Source badge ── */
.source-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:700; margin-bottom:8px;
    letter-spacing:.05em; text-transform:uppercase;
}

/* ── Prediction bars ── */
.pred-bar-wrap { margin:7px 0; }
.pred-bar-label {
    display:flex; justify-content:space-between;
    font-size:0.82rem; margin-bottom:3px;
    font-weight:500; color:#1a1a1a;
}
.pred-bar-bg {
    background:#e2ede6; border-radius:6px;
    height:13px; width:100%; overflow:hidden;
}
.pred-bar-fill { height:13px; border-radius:6px; }

/* ── Buttons ── */
div.stButton > button {
    background:linear-gradient(135deg,#1a5c35,#2e8b57);
    color:white; border:none; border-radius:10px;
    font-family:'Syne',sans-serif; font-weight:700;
    font-size:0.95rem; padding:0.6rem 1.5rem;
    width:100%; transition:all .25s;
}
div.stButton > button:hover {
    background:linear-gradient(135deg,#2e8b57,#3aad6e);
    transform:translateY(-2px);
    box-shadow:0 6px 20px rgba(46,139,87,.35);
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom:3px solid #2e8b57 !important;
    color:#1a5c35 !important; font-weight:700;
}
.stSuccess { border-left:4px solid #2e8b57 !important; }
.stWarning { border-left:4px solid #f0a500 !important; }
.stInfo    { border-left:4px solid #3498db !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  ENV & API
# ══════════════════════════════════════════════
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    st.error("🔑 **GROQ_API_KEY** not found. Add it to your `.env` file.")
    st.stop()

client       = Groq(api_key=GROQ_KEY)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHAT_MODEL   = "llama-3.3-70b-versatile"
CONF_THRESH  = 0.45

# ══════════════════════════════════════════════
#  LOAD LOCAL ML ASSETS
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model…")
def load_assets():
    if not os.path.exists('crop_model.h5'):
        raise FileNotFoundError("crop_model.h5 not found. Run `python train.py` first.")
    if not os.path.exists('labels.txt'):
        raise FileNotFoundError("labels.txt not found. Run `python train.py` first.")
    model = tf.keras.models.load_model('crop_model.h5')
    with open('labels.txt') as f:
        labels = [l.strip() for l in f if l.strip()]
    return model, labels

try:
    local_model, crop_labels = load_assets()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

# ══════════════════════════════════════════════
#  IMAGE ENHANCEMENT
# ══════════════════════════════════════════════
def enhance_image(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    mean_lum = np.array(img).mean()
    if mean_lum < 70:
        img = ImageEnhance.Brightness(img).enhance(1.7)
    elif mean_lum > 210:
        img = ImageEnhance.Brightness(img).enhance(0.75)
    img = ImageEnhance.Color(img).enhance(1.3)
    return img

# ══════════════════════════════════════════════
#  LOCAL ML PREDICTION (backup)
# ══════════════════════════════════════════════
def local_predict(img: Image.Image):
    def _run(i):
        i   = i.convert("RGB").resize((224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(i) / 255.0
        arr = np.expand_dims(arr, 0)
        return local_model.predict(arr, verbose=0)[0]

    probs = _run(img)
    conf  = float(np.max(probs))

    if conf < CONF_THRESH:
        probs2 = _run(enhance_image(img))
        if float(np.max(probs2)) > conf:
            probs, conf = probs2, float(np.max(probs2))

    top5 = [(crop_labels[i], round(float(probs[i]) * 100, 1))
            for i in np.argsort(probs)[::-1][:5]]

    return crop_labels[int(np.argmax(probs))], round(conf * 100, 1), top5

# ══════════════════════════════════════════════
#  GROQ VISION — PRIMARY IDENTIFIER
#  Much more accurate than local model
# ══════════════════════════════════════════════
def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def groq_identify(image: Image.Image) -> tuple:
    """Use Groq Vision to identify the crop.
    Returns (label, confidence, reason, top5_list)
    top5_list = [(name, confidence), ...]
    """
    b64       = encode_image(image)
    supported = ", ".join(crop_labels)

    prompt = f"""Look at this image carefully and identify the crop/plant shown.

Choose ONLY from this list: {supported}

Reply in this EXACT format (nothing else, no extra text):
CROP: <best match from the list>
CONFIDENCE: <number 1-100>
REASON: <one sentence why>
ALT1: <2nd most likely crop from list>
ALT1_CONF: <confidence 1-100>
ALT2: <3rd most likely crop from list>
ALT2_CONF: <confidence 1-100>
ALT3: <4th most likely crop from list>
ALT3_CONF: <confidence 1-100>
ALT4: <5th most likely crop from list>
ALT4_CONF: <confidence 1-100>

If you cannot identify any crop, use "unknown" for CROP."""

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        max_tokens=200
    )

    text  = resp.choices[0].message.content.strip()
    data  = {}
    for line in text.split('\n'):
        if ':' in line:
            key, _, val = line.partition(':')
            data[key.strip()] = val.strip()

    def _match(raw):
        if not raw or raw.lower() == "unknown":
            return "unknown"
        for c in crop_labels:
            if c.lower() == raw.lower():
                return c
        for c in crop_labels:
            if raw.lower() in c.lower() or c.lower() in raw.lower():
                return c
        return raw   # keep as-is if no match

    def _conf(key):
        try:    return max(0, min(100, int(data.get(key, 0))))
        except: return 0

    label      = _match(data.get("CROP", "unknown"))
    confidence = _conf("CONFIDENCE")
    reason     = data.get("REASON", "")

    # Build top-5 from Groq's alternatives
    top5 = [(label, confidence)]
    for i in range(1, 5):
        alt_name = _match(data.get(f"ALT{i}", ""))
        alt_conf = _conf(f"ALT{i}_CONF")
        if alt_name and alt_name != "unknown":
            top5.append((alt_name, alt_conf))

    # Pad to 5 if needed
    while len(top5) < 5:
        top5.append(("—", 0))

    return label, confidence, reason, top5

# ══════════════════════════════════════════════
#  GROQ DIET REPORT
# ══════════════════════════════════════════════
def groq_diet_report(image: Image.Image, label: str) -> str:
    b64 = encode_image(image)
    prompt = f"""The crop identified is **{label}**.

Generate a comprehensive **Human Diet & Nutrition Report**:

## 1. 🌾 Crop Overview
Brief description and culinary significance.

## 2. 📊 Nutritional Profile (per 100 g)
Table: calories, carbs, protein, fat, fibre, key vitamins & minerals.

## 3. 💪 Top Health Benefits
At least 6 evidence-based benefits with explanations.

## 4. 📅 Recommended Daily Intake
For: Children · Teenagers · Adults · Elderly · Pregnant women

## 5. 🍽️ Best Ways to Consume
Raw vs cooked vs processed — which preserves most nutrients.

## 6. 📆 Sample Weekly Diet Plan
7-day plan (Mon–Sun) with Breakfast / Lunch / Dinner using **{label}**.

## 7. ⚠️ Who Should Limit or Avoid
Groups who should be careful.

Focus ONLY on human nutrition. Do NOT mention soil, water, or farming.
"""
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        max_tokens=2048
    )
    return resp.choices[0].message.content

# ══════════════════════════════════════════════
#  URL IMAGE LOADER — fixed with SSL + headers
# ══════════════════════════════════════════════
def load_image_from_url(url: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        return img, None
    except Exception as e1:
        try:
            # Fallback: urllib
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0"
            })
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
                img = Image.open(BytesIO(r.read()))
                return img, None
        except Exception as e2:
            return None, f"Could not load image: {e2}"

# ══════════════════════════════════════════════
#  IMAGE HASH
# ══════════════════════════════════════════════
def image_hash(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").resize((64, 64)).save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
defaults = {
    "chat_history": [], "report": "", "id_name": "",
    "confidence": 0, "is_unclear": False, "top5": [],
    "last_image_hash": "", "id_source": "", "id_reason": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════
#  PREDICTION BARS
# ══════════════════════════════════════════════
def render_pred_bars(top5):
    colors = ["#1a5c35","#2e8b57","#3aad6e","#52c788","#74d9a0"]
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    html   = "<div style='margin-top:8px'>"
    for i, (name, pct) in enumerate(top5):
        c = colors[i]
        html += f"""
        <div class='pred-bar-wrap'>
          <div class='pred-bar-label'>
            <span>{medals[i]}&nbsp;<b>{name}</b></span>
            <span style='color:{c};font-weight:700'>{pct}%</span>
          </div>
          <div class='pred-bar-bg'>
            <div class='pred-bar-fill' style='width:{min(pct,100)}%;background:{c};'></div>
          </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 CropID")
    st.markdown("**AI-Powered Crop Identification & Nutrition System**")
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("1. 📸 Upload / URL / Webcam\n2. 🚀 Click **Identify**\n3. 📋 View report\n4. 💬 Ask chatbot")
    st.markdown("---")
    st.markdown("### 🤖 Identification Mode")
    st.markdown("**Primary:** Groq Vision AI *(very accurate)*")
    st.markdown("**Backup:** Local ML Model")
    st.markdown("---")
    st.markdown("### Supported Crops")
    for c in crop_labels:
        st.markdown(f"- {c}")
    st.markdown("---")
    st.caption("TensorFlow · Groq LLaMA · Streamlit")

# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:10px 0 20px'>
  <h1 style='font-family:Syne,sans-serif;font-size:2.4rem;
    background:linear-gradient(135deg,#1a5c35,#2e8b57);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;'>
    🌿 CropID Multimodal System
  </h1>
  <p style='color:#555;font-size:1rem;'>
    Identify any crop from a photo — get a full nutrition &amp; diet report
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Your Crop Image")
    source = st.radio("Source:", ["Upload File", "Image URL", "Webcam"],
                      horizontal=True, label_visibility="collapsed")

    img = None

    if source == "Upload File":
        f = st.file_uploader("Upload", type=['jpg','jpeg','png','webp'],
                             label_visibility="collapsed")
        if f:
            img = Image.open(f)

    elif source == "Image URL":
        url = st.text_input("Paste any image URL here:",
                            placeholder="https://example.com/crop.jpg")
        if url:
            with st.spinner("Loading image from URL…"):
                img, err = load_image_from_url(url)
            if err:
                st.error(f"❌ {err}\n\nTip: Try downloading the image and uploading it directly.")
            else:
                st.success("✅ Image loaded successfully!")

    elif source == "Webcam":
        cam = st.camera_input("Take a photo")
        if cam:
            img = Image.open(cam)

    if img:
        # Auto-clear on new image
        h = image_hash(img)
        if h != st.session_state.last_image_hash:
            st.session_state.chat_history    = []
            st.session_state.last_image_hash = h
            st.session_state.id_name         = ""
            st.session_state.report          = ""
            st.session_state.top5            = []

        st.image(img, use_column_width=True,
                 caption="Uploaded image — ready for analysis")

        if st.button("🚀  IDENTIFY CROP & GENERATE DIET REPORT"):
            with st.spinner("🔍 Running AI analysis…"):

                # ── Step 1: Groq Vision (primary — very accurate) ──
                groq_label, groq_conf, groq_reason, groq_top5 = groq_identify(img)

                # ── Step 2: Local ML (fallback only) ──
                local_label, local_conf, _ = local_predict(img)

                # ── Step 3: Pick best result ──
                if groq_label != "unknown" and groq_conf > 0:
                    final_label  = groq_label
                    final_conf   = groq_conf
                    id_source    = "🤖 Groq Vision AI"
                    is_unclear   = groq_conf < 50
                    top5         = groq_top5   # use Groq alternatives
                else:
                    # Fallback to local model
                    final_label  = local_label
                    final_conf   = local_conf
                    id_source    = "🧠 Local ML Model"
                    is_unclear   = local_conf < 45
                    top5         = []

                # ── Step 4: Generate diet report ──
                report = groq_diet_report(img, final_label)

                st.session_state.id_name      = final_label
                st.session_state.confidence   = final_conf
                st.session_state.is_unclear   = is_unclear
                st.session_state.top5         = top5
                st.session_state.report       = report
                st.session_state.id_source    = id_source
                st.session_state.id_reason    = groq_reason
                st.session_state.chat_history = []

            st.rerun()

    # ── Results ──
    if st.session_state.id_name:
        st.markdown("---")

        conf       = st.session_state.confidence
        is_unclear = st.session_state.is_unclear
        conf_color = "#c0392b" if is_unclear else ("#e67e22" if conf < 70 else "#1a7a42")
        status_txt = "⚠️ Unclear" if is_unclear else "✅ Clear"

        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card' style='background:linear-gradient(135deg,#1a5c35,#2e8b57)'>
            <div class='m-val'>{st.session_state.id_name}</div>
            <div class='m-lbl'>Identified Crop</div>
          </div>
          <div class='metric-card' style='background:{conf_color}'>
            <div class='m-val'>{conf}%</div>
            <div class='m-lbl'>Confidence</div>
          </div>
          <div class='metric-card' style='background:#2c3e50'>
            <div class='m-val'>{status_txt}</div>
            <div class='m-lbl'>Image Quality</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Source badge
        badge_color = "#1a5c35" if "Groq" in st.session_state.id_source else "#7f8c8d"
        st.markdown(f"""
        <span class='source-badge' style='background:{badge_color};color:white'>
          {st.session_state.id_source}
        </span>
        """, unsafe_allow_html=True)

        if st.session_state.id_reason:
            st.caption(f"💡 {st.session_state.id_reason}")

        if is_unclear:
            st.warning("⚠️ **Low confidence** — try a clearer, well-lit photo.")

        tab_id, tab_diet = st.tabs(["🆔 Identification", "🥗 Diet & Nutrition Report"])

        with tab_id:
            st.markdown("#### 📊 Top 5 Predictions (Groq Vision AI)")
            if st.session_state.top5:
                render_pred_bars(st.session_state.top5)
            st.info("💡 **Groq Vision AI** identifies the crop directly from the image — highly accurate.")

        with tab_diet:
            if st.session_state.report:
                st.markdown(st.session_state.report)

# ── RIGHT COLUMN — Chatbot ──
with col_right:
    st.markdown("### 💬 Crop & Nutrition Expert Chat")
    st.caption("Ask about nutrition, recipes, diet plans, or health benefits.")

    chat_box = st.container(height=520)
    for msg in st.session_state.chat_history:
        with chat_box.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.id_name:
        st.info("👈 Identify a crop first, then ask the expert!")

    if q := st.chat_input("e.g. Is this good for diabetics? Best recipes?"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        with chat_box.chat_message("user"):
            st.markdown(q)

        with chat_box.chat_message("assistant"):
            system = (
                f"You are a certified nutritionist and agricultural expert. "
                f"Identified crop: **{st.session_state.id_name}** "
                f"(confidence: {st.session_state.confidence}%, "
                f"source: {st.session_state.id_source}). "
                f"Diet report:\n\n{st.session_state.report}\n\n"
                "Answer concisely. Focus on diet, nutrition, recipes, health only."
            )
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    *[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.chat_history],
                ],
                max_tokens=1024
            )
            ans = resp.choices[0].message.content
            st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()