import logging
import os
import sys
import json
from pathlib import Path
import base64

# Streamlit sets sys.path[0] to this folder; project root must be on path for `import dashboard`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if not logging.root.handlers:
    _lvl = getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    logging.basicConfig(level=_lvl, format="%(levelname)s %(name)s: %(message)s")

import streamlit as st
from dashboard.views import ml_strategy, stock_analysis

st.set_page_config(page_title="StockPulse Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- UI Config ---
ui_config = {"brand_name": "StockPulse", "tagline": "AI &middot; QUANT &middot; RESEARCH"}
config_path = Path(__file__).parent / "ui_config.json"
if config_path.exists():
    try:
        with open(config_path, "r") as f:
            ui_config = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load ui_config.json: {e}")

# --- Global Design System ---
from dashboard.styles.tokens import CSS_TOKENS
st.markdown(CSS_TOKENS, unsafe_allow_html=True)

# --- Reusable Neon Card Component ---
def render_neon_card(title, value):
    return f"""<div class="neon-card">
<div class="neon-card-inner">
<div class="neon-card-title">{title}</div>
<div class="neon-card-value">{value}</div>
</div>
</div>"""

# Add it to session state so other modules can use it
if "render_neon_card" not in st.session_state:
    st.session_state.render_neon_card = render_neon_card

# --- Sidebar & Navigation ---
logo_svg_path = Path(__file__).parent / "assets/logo.svg"
logo_png_path = Path(__file__).parent / ui_config.get('logo_path', 'assets/logo.png')

brand_name = ui_config.get("brand_name", "StockPulse")
tagline = ui_config.get("tagline", "AI &middot; QUANT &middot; RESEARCH")

logo_img_tag = ""
if logo_svg_path.exists():
    with open(logo_svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    encoded_svg = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    logo_img_tag = f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="80" height="80" style="border-radius: 14px; object-fit: contain;">'
elif logo_png_path.exists():
    with open(logo_png_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    logo_img_tag = f'<img src="data:image/png;base64,{encoded_string}" width="80" height="80" style="border-radius: 14px; object-fit: contain;">'

st.sidebar.markdown(f"""
<div style="text-align:center; padding: 16px 0 8px 0;">
    <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0; margin-bottom:4px;">{brand_name}</div>
    <div style="font-size:0.7rem; color:#00d4ff; letter-spacing:0.12em; margin-bottom:14px;">{tagline}</div>
    {logo_img_tag}
</div>
<hr style="border:none; border-top:1px solid rgba(255,255,255,0.07); margin: 12px 0 16px 0;">
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Stock Analysis", "ML Strategy"])

st.sidebar.markdown("<hr class='gradient-divider'>", unsafe_allow_html=True)

if page == "Stock Analysis":
    stock_analysis.show()
elif page == "ML Strategy":
    ml_strategy.show()