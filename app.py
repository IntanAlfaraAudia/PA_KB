
import base64
import io
import json
import os
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "berry_cnn.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = 128                 # sama dengan IMG_SIZE saat training
LOW_CONF = 0.60                # saran dari analisis error di notebook
HIGH_CONF = 0.80
GRADCAM_LAYER = "last_conv_relu"

# Ubah sesuai kelompok Anda
TEAM_NAME = "Proyek Akhir Praktikum Kecerdasan Buatan"

st.set_page_config(
    page_title="BerryVision | Klasifikasi Berry",
    page_icon="🫐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Informasi tiap kelas (kunci = nama kelas persis seperti di class_names.json)
BERRY_INFO = {
    "Tayberry": {
        "label": "Tayberry",
        "emoji": "🍓",
        "color": "#ff5d8f",
        "latin": "Rubus fruticosus × R. idaeus",
        "desc": "Silangan blackberry dan raspberry. Buahnya memanjang, berwarna merah tua keunguan saat matang.",
        "ciri": ["Buah lonjong, merah tua keunguan", "Permukaan berbulir seperti raspberry", "Rasa manis-asam yang kuat"],
    },
    "Wineberry": {
        "label": "Wineberry",
        "emoji": "🍒",
        "color": "#ff3b4e",
        "latin": "Rubus phoenicolasius",
        "desc": "Berry liar asal Asia Timur dengan buah kecil berwarna merah cerah dan berbulir halus.",
        "ciri": ["Merah oranye sampai merah cerah", "Berbulir, sering tampak mengkilap", "Kelopak dan batang berbulu kemerahan"],
    },
    "black berry": {
        "label": "Blackberry",
        "emoji": "🫐",
        "color": "#a78bfa",
        "latin": "Rubus fruticosus (agg.)",
        "desc": "Buah majemuk berwarna hitam mengkilap yang tumbuh pada semak berduri.",
        "ciri": ["Hitam pekat mengkilap", "Tersusun dari banyak butir kecil", "Bentuk bulat memanjang"],
    },
    "blueberry": {
        "label": "Blueberry",
        "emoji": "🔵",
        "color": "#38bdf8",
        "latin": "Vaccinium spp.",
        "desc": "Buah bulat kecil berwarna biru keunguan dengan lapisan lilin keputihan di permukaannya.",
        "ciri": ["Bulat, biru keunguan", "Lapisan lilin keputihan (bloom)", "Ada mahkota kecil di ujung buah"],
    },
    "white mulberry": {
        "label": "White Mulberry",
        "emoji": "🤍",
        "color": "#fcd34d",
        "latin": "Morus alba",
        "desc": "Buah pohon murbei putih. Warnanya putih kekuningan sampai merah muda atau ungu saat matang.",
        "ciri": ["Putih, krem, atau kehijauan", "Bisa memerah muda saat matang", "Bentuk lonjong kecil bergerombol"],
    },
}

# Hasil evaluasi dari notebook (data test, 378 gambar)
TEST_ACC, TEST_F1, TEST_LOSS = 88.36, 87.83, 0.4099
PER_CLASS = [
    ("Tayberry", 0.892, 0.773, 0.829, 75),
    ("Wineberry", 0.705, 0.893, 0.788, 75),
    ("black berry", 0.955, 0.977, 0.966, 87),
    ("blueberry", 0.966, 0.925, 0.945, 93),
    ("white mulberry", 0.950, 0.792, 0.864, 48),
]
ROBUSTNESS = [
    ("Normal", 87.83),
    ("Noise gaussian", 83.60),
    ("JPEG q=15", 84.13),
    ("Rotasi 30°", 83.33),
    ("Gelap", 80.95),
    ("Terang", 79.37),
    ("Blur", 52.12),
]


def info_for(name: str) -> dict:
    """Ambil info kelas; fallback aman jika nama kelas tidak dikenal."""
    return BERRY_INFO.get(
        name,
        {"label": name.title(), "emoji": "🍇", "color": "#c084fc", "latin": "-",
         "desc": "", "ciri": []},
    )


# --------------------------------------------------------------------------- #
# Gaya (CSS)
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(255,77,141,.16), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(124,58,237,.18), transparent 55%),
        #0e0a14;
}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden;
    border-radius: 28px; padding: 38px 40px 32px;
    background: linear-gradient(135deg, #2b0f3f 0%, #4a1456 45%, #7a1a4d 100%);
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 24px 60px -20px rgba(255,77,141,.35);
}
.hero:before, .hero:after {
    content: ""; position: absolute; border-radius: 50%; filter: blur(2px); opacity: .55;
}
.hero:before {width: 260px; height: 260px; right: -60px; top: -90px;
    background: radial-gradient(circle at 30% 30%, #ff8fb7, #ff4d8d 55%, transparent 70%);}
.hero:after {width: 180px; height: 180px; right: 150px; bottom: -90px;
    background: radial-gradient(circle at 30% 30%, #a78bfa, #7c3aed 55%, transparent 70%);}
.hero .eyebrow {
    display: inline-block; font-size: .74rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #ffd1e3; background: rgba(255,255,255,.10);
    padding: 6px 12px; border-radius: 999px; margin-bottom: 14px;
}
.hero h1 {
    margin: 0 0 8px; font-size: clamp(2rem, 4.2vw, 3.1rem); line-height: 1.05;
    font-weight: 800; letter-spacing: -.02em; color: #fff;
}
.hero h1 span {
    background: linear-gradient(90deg, #ffb3cf, #ff6fa4 50%, #c4b5fd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p {margin: 0; max-width: 640px; color: #eadcf5; font-size: 1.02rem; line-height: 1.6;}
.chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; position: relative; z-index: 1;}
.chip {
    display: inline-flex; align-items: center; gap: 8px; padding: 7px 13px;
    border-radius: 999px; font-size: .84rem; font-weight: 600; color: #fff;
    background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.14);
    backdrop-filter: blur(6px);
}
.dot {width: 9px; height: 9px; border-radius: 50%; display: inline-block;}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid rgba(255,255,255,.08);}
.stTabs [data-baseweb="tab"] {
    height: 46px; padding: 0 18px; border-radius: 12px 12px 0 0;
    font-weight: 600; color: #b9aecb;
}
.stTabs [aria-selected="true"] {color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"] {background: #ff4d8d; height: 3px;}

/* ---------- Kartu ---------- */
.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    border-radius: 22px; padding: 22px 24px; margin-bottom: 18px;
    backdrop-filter: blur(8px);
}
.card h3 {margin: 0 0 4px; font-size: 1.05rem; font-weight: 700; color: #fff;}
.card .sub {color: #a99bbd; font-size: .88rem; margin-bottom: 14px;}
.sec-title {
    font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    color: #a99bbd; margin: 6px 0 12px;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.035); border: 1.5px dashed rgba(255,77,141,.45);
    border-radius: 18px; padding: 26px 18px;
}
[data-testid="stFileUploaderDropzone"]:hover {border-color: #ff4d8d; background: rgba(255,77,141,.07);}
[data-testid="stCameraInput"] > div {border-radius: 18px; overflow: hidden;}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stCameraInput"] button {border-radius: 12px;}

/* ---------- Foto ---------- */
.photo {
    width: 100% !important; height: auto !important; object-fit: contain !important;
    border-radius: 18px; display: block;
    border: 1px solid rgba(255,255,255,.10); box-shadow: 0 14px 34px -14px rgba(0,0,0,.7);
}
.photo-cap {font-size: .8rem; color: #a99bbd; margin: 8px 2px 0; text-align: center;}

/* ---------- Hasil ---------- */
.result {
    position: relative; overflow: hidden; border-radius: 24px; padding: 26px 26px 22px;
    border: 1px solid rgba(255,255,255,.10);
    background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
}
.result .glow {
    position: absolute; width: 260px; height: 260px; right: -80px; top: -110px;
    border-radius: 50%; opacity: .35; filter: blur(30px);
}
.result .kicker {font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #a99bbd;}
.result .name {
    display: flex; align-items: center; gap: 14px; margin: 8px 0 2px;
    font-size: clamp(1.8rem, 3.4vw, 2.5rem); font-weight: 800; letter-spacing: -.02em; color: #fff;
}
.result .emoji {
    width: 58px; height: 58px; border-radius: 18px; display: grid; place-items: center;
    font-size: 1.9rem; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12);
}
.result .latin {color: #b9aecb; font-style: italic; font-size: .92rem;}
.conf-row {display: flex; align-items: baseline; justify-content: space-between; margin-top: 20px;}
.conf-num {font-size: 2.3rem; font-weight: 800; letter-spacing: -.03em; color: #fff;}
.conf-num small {font-size: 1.1rem; color: #b9aecb; font-weight: 700;}
.badge {
    font-size: .8rem; font-weight: 700; padding: 6px 12px; border-radius: 999px;
    border: 1px solid transparent;
}
.badge.ok   {color: #6ee7b7; background: rgba(16,185,129,.14); border-color: rgba(16,185,129,.35);}
.badge.mid  {color: #fcd34d; background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.35);}
.badge.low  {color: #fda4af; background: rgba(244,63,94,.14); border-color: rgba(244,63,94,.38);}
.meter {height: 10px; border-radius: 99px; background: rgba(255,255,255,.09); overflow: hidden; margin-top: 10px;}
.meter > div {height: 100%; border-radius: 99px;}

/* ---------- Bar peluang ---------- */
.prob {margin: 13px 0;}
.prob .top {display: flex; justify-content: space-between; font-size: .92rem; font-weight: 600; color: #e9e1f3; margin-bottom: 6px;}
.prob .top span:last-child {color: #fff; font-variant-numeric: tabular-nums;}
.prob .bar {height: 9px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden;}
.prob .bar > div {height: 100%; border-radius: 99px;}
.prob.first .top {font-weight: 800;}

/* ---------- Callout & chip kualitas ---------- */
.callout {
    display: flex; gap: 12px; padding: 14px 16px; border-radius: 16px; margin: 12px 0;
    font-size: .92rem; line-height: 1.55; color: #eadff7;
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
}
.callout .ic {font-size: 1.15rem; line-height: 1.4;}
.callout.warn {background: rgba(245,158,11,.09); border-color: rgba(245,158,11,.30);}
.callout.bad  {background: rgba(244,63,94,.09); border-color: rgba(244,63,94,.32);}
.callout.good {background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.28);}
.callout b {color: #fff;}
.q-row {display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 2px;}
.q {
    font-size: .8rem; font-weight: 600; padding: 6px 11px; border-radius: 999px;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); color: #d9cfe8;
}
.q.good {color: #6ee7b7; border-color: rgba(16,185,129,.35);}
.q.warn {color: #fcd34d; border-color: rgba(245,158,11,.38);}
.q.bad  {color: #fda4af; border-color: rgba(244,63,94,.4);}

/* ---------- Kosong ---------- */
.empty {
    text-align: center; padding: 46px 26px; border-radius: 24px;
    border: 1px dashed rgba(255,255,255,.16); background: rgba(255,255,255,.025);
}
.empty .big {font-size: 3.2rem; margin-bottom: 8px;}
.empty h3 {margin: 0 0 6px; font-size: 1.2rem; color: #fff;}
.empty p {margin: 0 auto; max-width: 400px; color: #a99bbd; font-size: .93rem; line-height: 1.6;}
.tips {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 22px; text-align: left;}
.tip {padding: 14px; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);}
.tip b {display: block; color: #fff; font-size: .9rem; margin: 4px 0 2px;}
.tip span {color: #a99bbd; font-size: .8rem; line-height: 1.45;}

/* ---------- Metrik ---------- */
.metrics {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px;}
.metric {padding: 18px 18px 16px; border-radius: 20px; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09);}
.metric .v {font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; color: #fff;}
.metric .l {font-size: .8rem; color: #a99bbd; font-weight: 600; margin-top: 2px;}

table.tbl {width: 100%; border-collapse: collapse; font-size: .92rem;}
table.tbl th {text-align: left; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: #a99bbd; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.1);}
table.tbl td {padding: 11px 10px; border-bottom: 1px solid rgba(255,255,255,.06); color: #eadff7; font-variant-numeric: tabular-nums;}
table.tbl tr:last-child td {border-bottom: none;}
table.tbl td.num {text-align: right;}
table.tbl th.num {text-align: right;}

/* ---------- Kenali Berry ---------- */
.berry-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px;}
.berry-card {position: relative; overflow: hidden; padding: 22px; border-radius: 22px; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09);}
.berry-card .strip {position: absolute; left: 0; top: 0; bottom: 0; width: 5px;}
.berry-card h4 {margin: 0; font-size: 1.2rem; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 10px;}
.berry-card .latin {color: #a99bbd; font-style: italic; font-size: .86rem; margin: 2px 0 10px;}
.berry-card p {color: #d9cfe8; font-size: .92rem; line-height: 1.55; margin: 0 0 10px;}
.berry-card ul {margin: 0; padding-left: 18px; color: #c5b8d9; font-size: .88rem; line-height: 1.7;}

.footer {text-align: center; color: #85799a; font-size: .82rem; margin-top: 34px; line-height: 1.7;}
.footer a {color: #ff8fb7; text-decoration: none;}

@media (max-width: 760px) {
    .hero {padding: 28px 22px 24px;}
    .metrics {grid-template-columns: repeat(2, 1fr);}
    .tips {grid-template-columns: 1fr;}
}
</style>
"""


def html(s: str):
    """Render HTML (tanpa indentasi supaya tidak dianggap code block oleh markdown)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model CNN...")
def load_assets():
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    grad_model = None
    try:
        grad_model = keras.Model(
            model.inputs,
            [model.get_layer(GRADCAM_LAYER).output, model.output],
        )
    except Exception:
        grad_model = None  # Grad-CAM dinonaktifkan jika layer tidak ditemukan
    return model, grad_model, class_names


def to_model_input(img: Image.Image) -> np.ndarray:
    """Resize persis seperti saat training (tf.image.resize bilinear, skala 0-255).
    Rescaling 1/255 sudah ada di dalam model."""
    import tensorflow as tf

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    x = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE))  # float32, bilinear
    return x.numpy()[None, ...]


def predict(model, x: np.ndarray) -> np.ndarray:
    return model(x, training=False).numpy()[0]


def gradcam(grad_model, x: np.ndarray, class_idx: int) -> np.ndarray:
    """Grad-CAM untuk kelas class_idx, hasil dinormalisasi 0-1 (ukuran IMG_SIZE x IMG_SIZE)."""
    import tensorflow as tf

    xt = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        fmap, pred = grad_model(xt, training=False)
        score = pred[:, class_idx]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = Image.fromarray((cam * 255).astype("uint8")).resize(
        (IMG_SIZE, IMG_SIZE), Image.BICUBIC
    )
    return np.asarray(cam, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Utilitas gambar
# --------------------------------------------------------------------------- #
def load_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)  # perbaiki orientasi foto HP
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def to_data_uri(img: Image.Image, max_side: int = 720) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def jet(x: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = img.copy()
    base.thumbnail((720, 720))
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(base.size, Image.BICUBIC)
    c = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = jet(c) * 255.0
    out = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def quality_report(img: Image.Image) -> list[tuple[str, str, str]]:
    """Cek cepat kualitas foto. Blur adalah kelemahan terbesar model (uji robustness)."""
    g = np.asarray(img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32)
    lap = (
        -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    sharp = float(lap.var())          # kalibrasi: foto tajam median ~1500, hasil blur ~35
    bright = float(g.mean())
    w, h = img.size

    out = []
    if sharp < 150:
        out.append(("bad", "Foto buram", "blur"))
    elif sharp < 400:
        out.append(("warn", "Agak kurang tajam", "blur"))
    else:
        out.append(("good", "Foto tajam", "blur"))

    if bright < 60:
        out.append(("warn", "Terlalu gelap", "light"))
    elif bright > 205:
        out.append(("warn", "Terlalu terang", "light"))
    else:
        out.append(("good", "Pencahayaan baik", "light"))

    out.append(("warn" if min(w, h) < 96 else "good", f"{w}×{h}px", "size"))
    return out


# --------------------------------------------------------------------------- #
# Komponen UI
# --------------------------------------------------------------------------- #
def render_hero(class_names):
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{info_for(c)["color"]}"></span>'
        f'{escape(info_for(c)["label"])}</span>'
        for c in class_names
    )
    html(
        f"""
        <div class="hero">
          <span class="eyebrow">Deep Learning · Custom CNN</span>
          <h1>Berry<span>Vision</span></h1>
          <p>Unggah foto buah dan biarkan model CNN mengenali jenis berry-nya,
          lengkap dengan tingkat keyakinan dan peta area yang diperhatikan model (Grad-CAM).</p>
          <div class="chips">{chips}</div>
        </div>
        """
    )


def render_empty():
    html(
        """
        <div class="empty">
          <div class="big">🫐</div>
          <h3>Belum ada foto</h3>
          <p>Unggah foto atau ambil gambar lewat kamera di sebelah kiri untuk mulai mengklasifikasi.</p>
          <div class="tips">
            <div class="tip">🔍<b>Pastikan tajam</b><span>Foto buram membuat akurasi turun drastis.</span></div>
            <div class="tip">💡<b>Cahaya cukup</b><span>Hindari foto terlalu gelap atau silau.</span></div>
            <div class="tip">🎯<b>Buah di tengah</b><span>Buah terlihat jelas dan mengisi sebagian besar bingkai.</span></div>
          </div>
        </div>
        """
    )


def render_result(class_names, probs, blurry=False):
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    top_name = class_names[top]
    meta = info_for(top_name)
    conf = float(probs[top])

    if blurry:
        badge = '<span class="badge low">Kurang andal · foto buram</span>'
    elif conf >= HIGH_CONF:
        badge = '<span class="badge ok">Yakin</span>'
    elif conf >= LOW_CONF:
        badge = '<span class="badge mid">Cukup yakin</span>'
    else:
        badge = '<span class="badge low">Kurang yakin</span>'

    html(
        f"""
        <div class="result">
          <div class="glow" style="background:{meta['color']}"></div>
          <div class="kicker">Hasil prediksi</div>
          <div class="name"><div class="emoji">{meta['emoji']}</div>{escape(meta['label'])}</div>
          <div class="latin">{escape(meta['latin'])}</div>
          <div class="conf-row">
            <div class="conf-num">{conf*100:.1f}<small>%</small></div>
            {badge}
          </div>
          <div class="meter"><div style="width:{conf*100:.1f}%;background:linear-gradient(90deg,{meta['color']},#ffffff55)"></div></div>
        </div>
        """
    )

    # ---- catatan kontekstual ----
    runner = int(order[1])
    runner_name = class_names[runner]
    pair = {top_name, runner_name} == {"Tayberry", "Wineberry"}
    if blurry:
        html(
            """<div class="callout bad"><span class="ic">🌫️</span><div>
            <b>Foto terdeteksi buram.</b> Pada uji ketangguhan, akurasi model turun hingga
            ±36 poin untuk gambar buram, jadi hasil ini sebaiknya tidak dijadikan patokan.
            Ambil ulang foto dengan fokus yang tajam.</div></div>"""
        )
    elif conf < LOW_CONF:
        html(
            f"""<div class="callout bad"><span class="ic">⚠️</span><div>
            <b>Prediksi kurang dapat dipercaya.</b> Keyakinan di bawah {int(LOW_CONF*100)}%.
            Coba foto yang lebih tajam dengan buah yang jelas terlihat. Perlu diingat model ini hanya
            mengenal 5 jenis berry, jadi objek lain tetap akan dipaksa masuk ke salah satunya.</div></div>"""
        )
    elif pair and probs[runner] >= 0.25:
        html(
            f"""<div class="callout warn"><span class="ic">🔁</span><div>
            <b>Tayberry dan Wineberry mirip secara visual.</b> Model ragu di antara keduanya
            ({escape(info_for(runner_name)['label'])} {probs[runner]*100:.0f}%). Pasangan ini memang
            kesalahan paling sering pada evaluasi model.</div></div>"""
        )
    elif probs[runner] >= 0.30:
        html(
            f"""<div class="callout warn"><span class="ic">🤔</span><div>
            Kandidat kedua cukup dekat: <b>{escape(info_for(runner_name)['label'])}</b>
            ({probs[runner]*100:.0f}%).</div></div>"""
        )

    # ---- peluang tiap kelas ----
    bars = ""
    for rank, i in enumerate(order):
        m = info_for(class_names[i])
        p = float(probs[i]) * 100
        bars += (
            f'<div class="prob {"first" if rank == 0 else ""}">'
            f'<div class="top"><span>{m["emoji"]} {escape(m["label"])}</span><span>{p:.1f}%</span></div>'
            f'<div class="bar"><div style="width:{max(p, 0.6):.1f}%;background:{m["color"]}"></div></div></div>'
        )
    html(f'<div class="card"><div class="sec-title">Peluang tiap kelas</div>{bars}</div>')


def render_gradcam(img, cam, class_names, class_idx):
    name = info_for(class_names[class_idx])["label"]
    html(
        f"""
        <div class="card">
          <h3>Grad-CAM: area yang diperhatikan model</h3>
          <div class="sub">Merah menandakan area yang paling memengaruhi prediksi
          <b>{escape(name)}</b>. Gunakan ini untuk memeriksa apakah model melihat buahnya,
          bukan latar belakang.</div>
        </div>
        """
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(f'<img class="photo" src="{to_data_uri(img)}"><div class="photo-cap">Foto asli</div>')
    with c2:
        html(f'<img class="photo" src="{to_data_uri(overlay_cam(img, cam))}"><div class="photo-cap">Overlay Grad-CAM</div>')


def render_about():
    html(
        f"""
        <div class="metrics">
          <div class="metric"><div class="v">{TEST_ACC:.2f}%</div><div class="l">Akurasi data test</div></div>
          <div class="metric"><div class="v">{TEST_F1:.2f}%</div><div class="l">Macro-F1</div></div>
          <div class="metric"><div class="v">2.519</div><div class="l">Gambar · 5 kelas</div></div>
          <div class="metric"><div class="v">{IMG_SIZE}×{IMG_SIZE}</div><div class="l">Ukuran input</div></div>
        </div>
        """
    )

    left, right = st.columns(2, gap="large")
    with left:
        rows = "".join(
            f'<tr><td><span class="dot" style="background:{info_for(n)["color"]};margin-right:8px"></span>'
            f'{escape(info_for(n)["label"])}</td><td class="num">{p:.3f}</td><td class="num">{r:.3f}</td>'
            f'<td class="num">{f:.3f}</td><td class="num">{s}</td></tr>'
            for n, p, r, f, s in PER_CLASS
        )
        html(
            f"""
            <div class="card">
              <h3>Performa per kelas</h3><div class="sub">Pada 378 gambar data test.</div>
              <table class="tbl">
                <tr><th>Kelas</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">N</th></tr>
                {rows}
              </table>
            </div>
            """
        )
    with right:
        bars = ""
        for n, a in ROBUSTNESS:
            color = "#ff4d8d" if a < 60 else ("#a78bfa" if n != "Normal" else "#34d399")
            bars += (
                f'<div class="prob"><div class="top"><span>{n}</span><span>{a:.1f}%</span></div>'
                f'<div class="bar"><div style="width:{a}%;background:{color}"></div></div></div>'
            )
        html(
            f"""
            <div class="card">
              <h3>Uji ketangguhan (robustness)</h3>
              <div class="sub">Akurasi saat foto test diberi gangguan.</div>
              {bars}
            </div>
            """
        )

    html(
        """
        <div class="callout warn"><span class="ic">💡</span><div>
        <b>Kelemahan utama: blur.</b> Akurasi turun dari ±88% menjadi ±52% pada gambar buram,
        kemungkinan karena model bertumpu pada tekstur halus permukaan buah. Itulah sebabnya
        aplikasi ini memeriksa ketajaman foto sebelum prediksi.</div></div>
        <div class="callout"><span class="ic">🧠</span><div>
        <b>Arsitektur:</b> Custom CNN dilatih dari nol (tanpa pre-trained weights): 4 blok konvolusi
        (Conv3×3 → BatchNorm → ReLU ×2, MaxPool, Dropout), Global Average Pooling, Dense + Dropout, dan
        softmax 5 kelas. Normalisasi piksel dan augmentasi berada di dalam model.</div></div>
        <div class="callout"><span class="ic">🔁</span><div>
        <b>Kelas yang sering tertukar:</b> Tayberry ↔ Wineberry (keduanya merah dan berbulir).
        Sebanyak 64% kesalahan model berakhir di kelas Wineberry.</div></div>
        """
    )


def render_guide(class_names):
    cards = ""
    for c in class_names:
        m = info_for(c)
        lis = "".join(f"<li>{escape(x)}</li>" for x in m["ciri"])
        cards += (
            f'<div class="berry-card"><div class="strip" style="background:{m["color"]}"></div>'
            f'<h4>{m["emoji"]} {escape(m["label"])}</h4><div class="latin">{escape(m["latin"])}</div>'
            f'<p>{escape(m["desc"])}</p><ul>{lis}</ul></div>'
        )
    html(f'<div class="berry-grid">{cards}</div>')
    html(
        """
        <div class="footer" style="margin-top:22px;text-align:left">
        Dataset: <a href="https://www.kaggle.com/datasets/srijitabandyopadhyay/berry-image-dataset" target="_blank">Berry Image Dataset (Kaggle)</a>
        oleh Bandopadhyay, Banerjee &amp; Debnath.</div>
        """
    )


# --------------------------------------------------------------------------- #
# Halaman utama
# --------------------------------------------------------------------------- #
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        st.error(
            "File model tidak ditemukan. Pastikan `berry_cnn.keras` dan `class_names.json` "
            "berada di folder yang sama dengan `app.py`."
        )
        st.stop()

    try:
        model, grad_model, class_names = load_assets()
    except Exception as e:  # noqa: BLE001
        st.error("Gagal memuat model. Pastikan versi TensorFlow/Keras sama dengan saat training.")
        st.exception(e)
        st.stop()

    render_hero(class_names)
    st.write("")

    tab_cls, tab_about, tab_guide = st.tabs(["🔍  Klasifikasi", "📊  Tentang Model", "🍓  Kenali Berry"])

    with tab_cls:
        left, right = st.columns([1, 1.12], gap="large")

        with left:
            html('<div class="sec-title">1 · Pilih foto</div>')
            t_up, t_cam = st.tabs(["Unggah foto", "Ambil foto"])
            with t_up:
                uploaded = st.file_uploader(
                    "Unggah foto berry",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    label_visibility="collapsed",
                )
            with t_cam:
                shot = st.camera_input("Ambil foto berry", label_visibility="collapsed")

            file = uploaded or shot
            img = None
            if file is not None:
                try:
                    img = load_image(file)
                except Exception:
                    st.error("Gambar tidak dapat dibaca. Coba file lain (JPG/PNG).")

            if img is not None:
                html(f'<img class="photo" src="{to_data_uri(img)}"><div class="photo-cap">Foto yang dianalisis</div>')
                chips = "".join(f'<span class="q {k}">{escape(t)}</span>' for k, t, _ in quality_report(img))
                html(f'<div class="q-row">{chips}</div>')
                show_cam = st.checkbox("Tampilkan Grad-CAM", value=True, disabled=grad_model is None)

        with right:
            html('<div class="sec-title">2 · Hasil</div>')
            if img is None:
                render_empty()
            else:
                with st.spinner("Menganalisis foto..."):
                    x = to_model_input(img)
                    probs = predict(model, x)
                blurry = any(level == "bad" and kind == "blur" for level, _, kind in quality_report(img))
                render_result(class_names, probs, blurry)

        if img is not None and grad_model is not None and show_cam:
            st.write("")
            cls_idx = int(np.argmax(probs))
            try:
                cam = gradcam(grad_model, x, cls_idx)
                render_gradcam(img, cam, class_names, cls_idx)
            except Exception:
                st.info("Grad-CAM tidak dapat dibuat untuk gambar ini.")

    with tab_about:
        st.write("")
        render_about()

    with tab_guide:
        st.write("")
        render_guide(class_names)

    html(
        f"""
        <div class="footer">
          {escape(TEAM_NAME)}<br>
          Model hanya mengenali 5 jenis berry. Hasil prediksi bersifat estimasi, bukan identifikasi pasti.
        </div>
        """
    )


main()
