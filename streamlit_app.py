import streamlit as st
import cv2
import torch
import numpy as np
import os

# --- YOUR LOCAL MODULES ---
from models.cnn_model import SimpleCNN
from utils.retina_validator import is_retinal_image

# =========================
#  PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Retinal Disease Detection",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="collapsed"
)

# =========================
#  GLOBAL STYLES (Clinical Dark Theme)
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}

    .stApp { background-color: #0b1121; color: #e2e8f0; }

    .hero {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 40px 20px;
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .hero h1 { margin: 0; padding-bottom: 10px; font-weight: 700; color: #38bdf8 !important; }
    .hero p { font-size: 1.2rem; opacity: 0.8; margin: 0; color: #94a3b8 !important;}

    .kpi-card {
        background-color: #1e293b;
        padding: 25px 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #334155;
        border-top: 4px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .kpi-title { color: #94a3b8; font-size: 1.1rem; font-weight: 500; margin-bottom: 10px; }
    .kpi-value { color: #38bdf8; font-size: 2.2rem; font-weight: 700; margin: 0; }
    
    .visual-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .visual-caption { color: #94a3b8; font-size: 0.95rem; font-weight: 500; margin-top: 10px; }

    .footer {
        margin-top: 60px;
        padding: 30px;
        border-radius: 16px;
        background: #0f172a;
        border: 1px solid #334155;
        text-align: center;
    }
    .footer h3 { color: #f8fafc !important; margin-bottom: 5px; font-weight: 600;}
    .footer p { color: #94a3b8 !important; font-size: 0.95rem; }
    .team-grid { margin-top: 15px; font-weight: 500; color: #cbd5e1 !important;}
    </style>
    """, unsafe_allow_html=True)

# =========================
# ⭐ MODEL LOADING
# =========================
@st.cache_resource(show_spinner="Loading diagnostic models...")
def load_model():
    try:
        model = SimpleCNN()
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load the model. Error: {e}")
        st.stop()

# =========================
# ⭐ MAIN APPLICATION
# =========================
def main():
    inject_custom_css()
    model = load_model()

    # --- HERO SECTION ---
    st.markdown("""
    <div class="hero">
        <h1>⚕️ Retinal Vascular Disease Detection</h1>
        <p>AI-Driven Fundus Image Analysis & Biomarker Extraction Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # --- UPLOAD SECTION ---
    st.markdown("### 📤 Upload Patient Scan")
    uploaded = st.file_uploader(
        "Upload a high-resolution retinal fundus image (.jpg, .jpeg, .png, .tif)",
        type=["jpg", "png", "jpeg", "tif"]
    )

    # --- PIPELINE ---
    if uploaded:
        temp_path = "temp_upload.jpg"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Validate Image
            if not is_retinal_image(temp_path):
                st.error("⚠️ The uploaded image does not appear to be a valid retinal fundus scan. Please verify the input.")
                st.stop()

            with st.spinner("🧠 Initializing neural network for vascular extraction..."):
                img = cv2.imread(temp_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                green = img_rgb[:,:,1]
                green = cv2.resize(green, (256, 256))
                green = green / 255.0
                tensor = torch.tensor(green).unsqueeze(0).unsqueeze(0).float()

                with torch.no_grad():
                    pred = model(tensor)

                pred_np = pred.squeeze().numpy()

                vessel_score = float(pred_np.mean())
                binary_mask = pred_np > 0.5
                vessel_density = float(binary_mask.sum() / binary_mask.size)
                vessel_pixels = int(binary_mask.sum())

            # --- DASHBOARD & KPI RENDERING ---
            st.markdown("---")
            st.markdown("### 📊 Biomarker Dashboard")

            k1, k2, k3 = st.columns(3)

            with k1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Mean Vessel Probability</div>
                    <div class="kpi-value">{vessel_score:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            with k2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Vessel Density</div>
                    <div class="kpi-value">{vessel_density:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with k3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Vessel Pixel Count</div>
                    <div class="kpi-value">{vessel_pixels:,}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("") 
            
            # =========================
            # ⭐ EXACT CRITICAL ALERT LOGIC
            # =========================
            if vessel_density < 0.08:
                st.error("⚠️ **CRITICAL ALERT:** Abnormal retinal vascular pattern detected. Vessel density is severely below the standard physiological threshold (8%).")
            else:
                st.success("✅ **ASSESSMENT:** Retinal vascular structure density appears within normal physiological parameters.")

            # --- VISUALIZATION PROCESSING ---
            st.markdown("### 🔬 Analysis Visualizations")
            
            h, w, _ = img_rgb.shape
            pred_up = cv2.resize(pred_np, (w, h))
            
            mask = np.zeros_like(img_rgb)
            mask[:, :, 1] = (pred_up * 255).astype(np.uint8) 
            mask[:, :, 2] = (pred_up * 255).astype(np.uint8) 
            overlay = cv2.addWeighted(img_rgb, 0.6, mask, 0.7, 0) 

            v1, v2, v3 = st.columns(3)

            with v1:
                st.markdown('<div class="visual-card">', unsafe_allow_html=True)
                st.image(img_rgb)
                st.markdown('<div class="visual-caption">1. Original Input Scan</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with v2:
                st.markdown('<div class="visual-card">', unsafe_allow_html=True)
                st.image(pred_up, clamp=True)
                st.markdown('<div class="visual-caption">2. AI Vessel Segmentation</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with v3:
                st.markdown('<div class="visual-card">', unsafe_allow_html=True)
                st.image(overlay)
                st.markdown('<div class="visual-caption">3. Pathological Overlay</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- TEAM FOOTER ---
    st.markdown("""
    <div class="footer">
        <h3>Malla Reddy University</h3>
        <p>B.Tech CSE – AI/ML (Zeta Batch)</p>
        <div class="team-grid">
            V. Shreyan (2211CS020660)<br>
            M. Mallika (2211CS020640) <br>
            R. Mourya (2211CS020685)
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()