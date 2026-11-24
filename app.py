import streamlit as st
from PIL import Image
import torch
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
import io
import base64

from utils.model_loader import load_model, load_class_names, DEFAULT_MODEL_PATH
from utils.preprocess import load_and_preprocess
from utils.predict import predict
from utils.gradcam import GradCAM, overlay_heatmap
from utils.disease_info import DISEASE_DATABASE, get_disease_details
from utils.leaf_validator import validate_leaf_presence, evaluate_prediction_confidence
from utils.icons import svg_icon

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LeafCare — Deteksi & Solusi Penyakit Daun Cabai",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# Load External Stylesheet (assets/style.css)
# ─────────────────────────────────────────────
css_path = Path(__file__).resolve().parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_data
def get_base64_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

hero_bg_path = Path(__file__).resolve().parent / "assets" / "bg_cabai.jpg"
if hero_bg_path.exists():
    b64 = get_base64_image(str(hero_bg_path))
    st.markdown(f"""
    <style>
    .agri-hero {{
        background-image:
            linear-gradient(105deg, rgba(23,54,40,0.92) 0%, rgba(23,54,40,0.75) 45%, rgba(23,54,40,0.30) 100%),
            url("data:image/jpeg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    @media (max-width: 768px) {{
        .agri-hero {{ background-position: 70% center; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Active Page Routing via Query Params
# ─────────────────────────────────────────────
current_page = st.query_params.get("page", "Studio Deteksi")
valid_pages = ["Studio Deteksi", "Ensiklopedia Penyakit", "Spesifikasi Model", "Tentang Kami"]
if current_page not in valid_pages:
    current_page = "Studio Deteksi"

# ─────────────────────────────────────────────
# Model & Resource Loader (Cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model MobileNetV2 & bobot terlatih...")
def get_model_bundle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, metadata = load_model(DEFAULT_MODEL_PATH, device=device)
    target_layer = model.features[-1]
    cam_extractor = GradCAM(model, target_layer)
    return model, class_names, metadata, cam_extractor, device

model, class_names, metadata, cam_extractor, device = get_model_bundle()

# ─────────────────────────────────────────────
# TOP NAVBAR (Editorial Clean Text Links - Clickable & Functional)
# ─────────────────────────────────────────────
nav_links_html = f"""
<div class="agri-navbar">
    <a href="?page=Studio+Deteksi" target="_self" class="agri-brand">
        <div class="agri-logo-icon">{svg_icon("leaf", size=24)}</div>
        <div class="agri-brand-text-col">
            <div class="agri-brand-title">LeafCare</div>
            <div class="agri-brand-subtitle">Grow Better. Protect Better.</div>
        </div>
    </a>
    <div class="agri-nav-menu">
        <a href="?page=Studio+Deteksi" target="_self" class="agri-nav-link {'active' if current_page == 'Studio Deteksi' else ''}">Studio Deteksi</a>
        <a href="?page=Ensiklopedia+Penyakit" target="_self" class="agri-nav-link {'active' if current_page == 'Ensiklopedia Penyakit' else ''}">Ensiklopedia Penyakit</a>
        <a href="?page=Spesifikasi+Model" target="_self" class="agri-nav-link {'active' if current_page == 'Spesifikasi Model' else ''}">Spesifikasi Model</a>
        <a href="?page=Tentang+Kami" target="_self" class="agri-nav-link {'active' if current_page == 'Tentang Kami' else ''}">Tentang Kami</a>
    </div>
    <a href="?page=Studio+Deteksi#studio-section" target="_self" class="agri-nav-cta">
        {svg_icon("sprout", size=16, color="#FFFFFF")}
        <span>Mulai Diagnosis</span>
    </a>
</div>
"""
st.markdown(nav_links_html, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# HALAMAN 1: STUDIO DETEKSI
# ═════════════════════════════════════════════
if current_page == "Studio Deteksi":
    # ─────────────────────────────────────────
    # HERO SECTION (Empowering Farmers. Growing Tomorrow.)
    # ─────────────────────────────────────────
    st.markdown(f"""
    <div class="agri-hero">
        <div>
            <div class="hero-tag">
                {svg_icon("sprout", size=14, color="#173628")}
                <span>Smart Farming for Chili Cultivation</span>
            </div>
            <div class="hero-heading">
                Empowering Farmers.<br>
                <span class="highlight">Growing Tomorrow.</span>
            </div>
            <p class="hero-description">
                LeafCare mengintegrasikan teknologi visi komputer presisi tinggi dan rekomendasi agronomi terarah 
                untuk mengidentifikasi penyakit daun cabai sejak dini, mengurangi risiko kerugian panen, 
                dan mewujudkan ketahanan pertanian yang berkelanjutan.
            </p>
            <div class="hero-actions">
                <a href="#studio-section" target="_self" class="btn-primary-pill">
                    <span>Mulai Deteksi Daun</span>
                    {svg_icon("leaf", size=16, color="#FFFFFF")}
                </a>
                <span class="btn-secondary-pill">
                    <span>Akurasi Uji: 94.22%</span>
                    {svg_icon("check-circle", size=15, color="#2D5F45")}
                </span>
            </div>
        </div>
        <div>
            <div class="hero-graphic-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="display:inline-flex; align-items:center; gap:0.4rem; background:rgba(255,255,255,0.12); backdrop-filter:blur(10px); border:1px solid rgba(255,255,255,0.2); padding:0.35rem 0.85rem; border-radius:20px; font-size:0.78rem; font-weight:600; color:#E2ECE5;">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#4ADE80;"></span>
                        Pemantauan Kebun Aktif
                    </span>
                    <span style="font-size:0.8rem; color:#A7C9B8;">Leaf Care</span>
                </div>
                <div style="margin-top: 1.2rem;">
                    <div style="font-family:'Playfair Display', serif; font-size:1.55rem; font-weight:700; margin:0 0 0.25rem 0; color:#FFFFFF;">
                        Sistem Diagnostik Daun Cabai
                    </div>
                    <p style="font-size:0.84rem; color:#C4DEC0; margin:0;">
                        MobileNetV2 Deep Learning dengan Guardrail Filter Non-Daun
                    </p>
                </div>
                <div class="hero-floating-glass">
                    <div class="glass-stat-item">
                        <div class="glass-stat-icon">{svg_icon("leaf", size=18, color="#FFFFFF")}</div>
                        <div>
                            <div class="glass-stat-label">Kondisi Daun</div>
                            <div class="glass-stat-value">Terverifikasi</div>
                        </div>
                    </div>
                    <div class="glass-stat-item">
                        <div class="glass-stat-icon">{svg_icon("shield-check", size=18, color="#FFFFFF")}</div>
                        <div>
                            <div class="glass-stat-label">Guardrail OOD</div>
                            <div class="glass-stat-value">Aktif</div>
                        </div>
                    </div>
                    <div class="glass-score-circle">
                        <div style="font-size:1.15rem; font-weight:800; color:#FFFFFF; line-height:1;">94%</div>
                        <div style="font-size:0.6rem; color:#D1E7DD; font-weight:600;">Akurasi</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # 4 FEATURES TILES (Smart Solutions for Modern Farming)
    # ─────────────────────────────────────────
    st.markdown(f"""
    <div class="section-center-head">
        <div class="section-label-sm">
            {svg_icon("sprout", size=14, color="#2D5F45")}
            <span>FITUR & LAYANAN UNGGULAN</span>
        </div>
        <div class="section-title-lg">Solusi Cerdas untuk Pertanian Modern</div>
    </div>

    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon-circle">{svg_icon("leaf", size=22, color="#173628")}</div>
            <div class="feature-card-title">Manajemen Penyakit</div>
            <p class="feature-card-desc">
                Klasifikasi presisi 6 kondisi daun cabai (Bercak Bakteri, Cercospora, Virus Keriting, Sehat, Defisiensi, dan Bercak Putih).
            </p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-circle">{svg_icon("microscope", size=22, color="#173628")}</div>
            <div class="feature-card-title">Visualisasi Grad-CAM</div>
            <p class="feature-card-desc">
                Peta panas transparan untuk melihat area spesifik daun yang menjadi dasar pertimbangan model saat memutuskan diagnosis.
            </p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-circle">{svg_icon("shield", size=22, color="#173628")}</div>
            <div class="feature-card-title">Guardrail Anti-Salah Tebak</div>
            <p class="feature-card-desc">
                Filter pintar berbasis spektrum klorofil dan morfologi untuk menolak gambar selain daun dan mencegah diagnosis asal.
            </p>
        </div>
        <div class="feature-card">
            <div class="feature-icon-circle">{svg_icon("file-text", size=22, color="#173628")}</div>
            <div class="feature-card-title">Rekomendasi Agronomi</div>
            <p class="feature-card-desc">
                Panduan penanganan terstruktur meliputi pengendalian kimiawi, biopestisida organik, dan pencegahan kultur teknis.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # STATISTICS EVERGREEN BANNER STRIP
    # ─────────────────────────────────────────
    st.markdown(f"""
    <div class="agri-stats-strip">
        <div class="stat-item-col">
            <div class="stat-strip-number">94.22%</div>
            <div class="stat-strip-label">Akurasi Pengujian Model</div>
        </div>
        <div class="stat-item-col">
            <div class="stat-strip-number">6 Kelas</div>
            <div class="stat-strip-label">Kondisi & Penyakit Daun</div>
        </div>
        <div class="stat-item-col">
            <div class="stat-strip-number">MobileNetV2</div>
            <div class="stat-strip-label">Transfer Learning CNN</div>
        </div>
        <div class="stat-item-col">
            <div class="stat-strip-number">&lt; 50 ms</div>
            <div class="stat-strip-label">Kecepatan Inferensi ({device.upper()})</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # STUDIO DETEKSI (Two-Column Workspace)
    # ─────────────────────────────────────────
    st.markdown('<div id="studio-section"></div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1.05, 1.35], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">1</span>
                <span>Pilih Sumber Gambar Daun</span>
            </div>
            """, unsafe_allow_html=True)

            input_mode = st.segmented_control(
                "Metode Input Gambar",
                options=["Unggah Berkas", "Sampel Uji", "Kamera Langsung"],
                default="Unggah Berkas",
                key="studio_input_mode",
                label_visibility="collapsed"
            )

            image_to_process = None
            source_label = ""

            if input_mode == "Unggah Berkas":
                uploaded_file = st.file_uploader(
                    "Pilih foto daun cabai (JPG, JPEG, PNG, WEBP)",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Pastikan foto memperlihatkan helai daun cabai secara jelas dengan pencahayaan yang cukup."
                )
                if uploaded_file is not None:
                    try:
                        image_to_process = Image.open(uploaded_file)
                        source_label = f"Berkas: {uploaded_file.name}"
                    except Exception as e:
                        st.error(f"Gagal membaca gambar: {e}")

            elif input_mode == "Sampel Uji":
                st.caption("Pilih gambar sampel untuk pengujian instan (termasuk sampel uji non-daun):")
                samples_dir = Path(__file__).resolve().parent / "assets" / "samples"
                
                curl_sample = samples_dir / "sampel_daun_keriting.jpeg"
                if not curl_sample.exists():
                    curl_sample = samples_dir / "sampel_daun_keriting.jpg"

                sample_options = {
                    "Daun Sehat (Healthy Leaf)": samples_dir / "sampel_daun_sehat.jpg",
                    "Bercak Daun Cercospora (Frog-Eye)": samples_dir / "sampel_bercak_daun.jpg",
                    "Defisiensi Nutrisi / Klorosis": samples_dir / "sampel_defisiensi_hara.jpg",
                    "Bercak Bakteri (Bacterial Spot)": samples_dir / "sampel_bercak_bakteri.jpg",
                    "Bercak Putih / White Spot": samples_dir / "sampel_bercak_putih.jpg",
                    "Daun Keriting / Curl Virus": curl_sample,
                    "[Uji Non-Daun] Objek Cangkir / Benda": samples_dir / "sampel_non_daun_benda.jpg",
                    "[Uji Non-Daun] Ilustrasi Wajah / Bukan Daun": samples_dir / "sampel_non_daun_wajah.jpg",
                }
                
                selected_sample = st.selectbox(
                    "Daftar Sampel",
                    options=list(sample_options.keys()),
                    label_visibility="collapsed"
                )
                
                sample_path = sample_options[selected_sample]
                if sample_path.exists():
                    image_to_process = Image.open(sample_path)
                    source_label = f"Sampel Uji: {selected_sample.split('(')[0].strip()}"
                else:
                    st.warning("File sampel uji tidak ditemukan di direktori assets/samples.")

            elif input_mode == "Kamera Langsung":
                camera_img = st.camera_input("Arahkan kamera ke daun cabai")
                if camera_img is not None:
                    try:
                        image_to_process = Image.open(camera_img)
                        source_label = "Tangkapan Kamera Langsung"
                    except Exception as e:
                        st.error(f"Gagal memproses tangkapan kamera: {e}")

            # Tampilkan Pratinjau Gambar Input di dalam Kartu yang Rapi
            if image_to_process is not None:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.markdown(f"**Pratinjau ({source_label})**")
                
                st.image(image_to_process, use_container_width=True, caption=f"Dimensi: {image_to_process.size[0]}×{image_to_process.size[1]} px")
                
                with st.expander("Pengaturan Visualisasi & Guardrail Daun", expanded=False):
                    cam_alpha = st.slider("Transparansi Heatmap Grad-CAM (Alpha)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
                    cam_cmap = st.selectbox("Skema Warna Heatmap", options=["JET", "TURBO", "VIRIDIS", "INFERNO"], index=0)
                    foliage_thresh = st.slider("Ambang Batas Minimal Jaringan Daun (%)", min_value=5, max_value=25, value=10, step=1,
                                               help="Persentase minimal spektrum klorofil/jaringan daun agar gambar dinyatakan valid.")
            else:
                st.markdown(f"""
                <div class="empty-placeholder-box">
                    <div class="empty-icon-circle">{svg_icon("leaf", size=24, color="#173628")}</div>
                    <div class="empty-title">Silakan Pilih atau Unggah Citra Daun</div>
                    <p class="empty-desc">
                        Gunakan berkas lokal, sampel pengujian siap pakai, atau kamera langsung untuk memulai diagnosis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                cam_alpha = 0.45
                cam_cmap = "JET"
                foliage_thresh = 10

    with col_result:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">2</span>
                <span>Hasil Diagnosis & Evaluasi Daun</span>
            </div>
            """, unsafe_allow_html=True)

            is_leaf = False

            if image_to_process is not None:
                # ─────────────────────────────────────────
                # Validasi Keberadaan Daun (OOD Filter)
                # ─────────────────────────────────────────
                is_leaf, val_code, val_reason, val_metrics = validate_leaf_presence(
                    image_to_process,
                    min_foliage_ratio=foliage_thresh / 100.0
                )

                if not is_leaf:
                    st.markdown(f"""
                    <div class="nonleaf-warning-card">
                        <span class="nonleaf-badge">
                            {svg_icon("shield", size=14, color="#9E2A12")}
                            <span>GUARDRAIL FILTER AKTIF • OBJEK BUKAN DAUN</span>
                        </span>
                        <div class="nonleaf-warning-title">
                            Objek Bukan Daun Cabai Terdeteksi
                        </div>
                        <p class="nonleaf-warning-desc">
                            Sistem mendeteksi bahwa gambar yang Anda masukkan <strong>bukan merupakan daun tanaman cabai</strong> 
                            atau berada di luar domain data latih model (<em>Out-of-Distribution</em>).
                        </p>
                        <div class="nonleaf-reason-box">
                            <strong>Alasan Penolakan:</strong> {val_reason}
                        </div>
                        <div class="nonleaf-guide-title">
                            Petunjuk Penggunaan yang Benar:
                        </div>
                        <ul class="nonleaf-guide-list">
                            <li>Pastikan gambar memperlihatkan helai daun tanaman cabai secara jelas dan fokus.</li>
                            <li>Hindari mengunggah foto manusia, hewan, kendaraan, benda mati, atau latar belakang kosong.</li>
                            <li>Gunakan pencahayaan alami yang merata tanpa bayangan ekstrem.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Rincian Metrik Analisis Citra (Mengapa Ditolak?)", expanded=True):
                        df_metrics = pd.DataFrame([
                            {"Indikator Citra": "Rasio Klorofil Hijau (HSV)", "Nilai Terdeteksi": f"{val_metrics.get('green_ratio', 0)}%", "Ambang Batas Minimum": "3.0%", "Keterangan": "Di bawah ambang batas (Bukan Daun)" if val_metrics.get('green_ratio', 0) < 3.0 else "Memenuhi"},
                            {"Indikator Citra": "Total Spektrum Jaringan Daun", "Nilai Terdeteksi": f"{val_metrics.get('foliage_ratio', 0)}%", "Ambang Batas Minimum": f"{foliage_thresh}%", "Keterangan": "Di bawah ambang batas" if val_metrics.get('foliage_ratio', 0) < foliage_thresh else "Memenuhi"},
                            {"Indikator Citra": "Dominasi Spektrum Warna Biru", "Nilai Terdeteksi": f"{val_metrics.get('blue_ratio', 0)}%", "Ambang Batas Maksimum": "35%", "Keterangan": "Melebihi batas (Bukan Daun)" if val_metrics.get('blue_ratio', 0) > 35 else "Normal"},
                            {"Indikator Citra": "Dominasi Spektrum Ungu/Magenta", "Nilai Terdeteksi": f"{val_metrics.get('purple_ratio', 0)}%", "Ambang Batas Maksimum": "35%", "Keterangan": "Melebihi batas (Bukan Daun)" if val_metrics.get('purple_ratio', 0) > 35 else "Normal"},
                            {"Indikator Citra": "Standar Deviasi Tekstur", "Nilai Terdeteksi": f"{val_metrics.get('texture_std', 0)}", "Ambang Batas Minimum": "10.0", "Keterangan": "Terlalu polos/seragam" if val_metrics.get('texture_std', 0) < 10 else "Normal"}
                        ])
                        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

                    with st.expander("Mode Evaluasi: Lihat Tebakan Mentah Model (Debug)", expanded=False):
                        st.caption("Peringatan: Model klasifikasi tertutup (softmax) akan selalu membagi probabilitas ke 6 kelas daun cabai meskipun objeknya bukan daun.")
                        img_size = metadata.get("img_size", 224)
                        norm_dict = metadata.get("normalize", None)
                        raw_tensor = load_and_preprocess(image_to_process, img_size=img_size, normalize_dict=norm_dict)
                        raw_results, _ = predict(model, raw_tensor, class_names, device=device)
                        st.write(f"**Tebakan Mentah:** `{raw_results['class_name']}` ({raw_results['confidence_pct']:.1f}%)")
                        st.dataframe(pd.DataFrame(raw_results["all_predictions"])[["rank", "class", "pct"]], hide_index=True)

                else:
                    img_size = metadata.get("img_size", 224)
                    normalize_dict = metadata.get("normalize", None)
                    input_tensor = load_and_preprocess(image_to_process, img_size=img_size, normalize_dict=normalize_dict)

                    results, pred_idx = predict(model, input_tensor, class_names, device=device)
                    top_class = results["class_name"]
                    top_conf = results["confidence_pct"]
                    latency = results["latency_ms"]
                    is_healthy = results["is_healthy"]

                    is_confident, conf_msg = evaluate_prediction_confidence(results, min_confidence=0.50)

                    disease_info = get_disease_details(top_class)
                    severity = disease_info["severity"]
                    severity_color = disease_info["severity_color"]
                    indo_name = disease_info["indonesian_name"]
                    category = disease_info["category"]

                    if not is_confident:
                        badge_bg = "#FFFBEB"
                        badge_border = "#FDE68A"
                        badge_color = "#92400E"
                        badge_icon = svg_icon("alert-triangle", size=18, color="#92400E")
                        banner_msg = f"Tingkat Keyakinan Rendah ({top_conf:.1f}%). {conf_msg}"
                    elif is_healthy:
                        badge_bg = "#F0FDF4"
                        badge_border = "#BBF7D0"
                        badge_color = "#166534"
                        badge_icon = svg_icon("check-circle", size=18, color="#166534")
                        banner_msg = "Tanaman Prima! Daun cabai terverifikasi sehat tanpa tanda-tanda infeksi berbahaya."
                    elif severity in ["Sangat Tinggi", "Tinggi"]:
                        badge_bg = "#FEF2F2"
                        badge_border = "#FECACA"
                        badge_color = "#991B1B"
                        badge_icon = svg_icon("alert-triangle", size=18, color="#991B1B")
                        banner_msg = f"Perhatian Diperlukan! Terindikasi penyakit berisiko tinggi ({category})."
                    else:
                        badge_bg = "#FFFBEB"
                        badge_border = "#FDE68A"
                        badge_color = "#92400E"
                        badge_icon = svg_icon("info", size=18, color="#92400E")
                        banner_msg = f"Terindikasi gejala {category}. Disarankan tindakan penanganan dini."

                    st.markdown(f"""
                    <div style="background:{badge_bg}; border:1px solid {badge_border}; border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:1.2rem; display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center; gap:0.75rem;">
                            <div>{badge_icon}</div>
                            <div>
                                <strong style="font-size:1.02rem; color:{badge_color};">{top_class} ({indo_name})</strong>
                                <div style="font-size:0.84rem; margin-top:0.15rem; color:{badge_color};">{banner_msg}</div>
                            </div>
                        </div>
                        <span style="background:{severity_color}; color:white; padding:0.32rem 0.8rem; border-radius:20px; font-size:0.75rem; font-weight:700;">
                            Risiko: {severity}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    conf_dot_color = "#2E7D32" if is_confident else "#F59E0B"
                    conf_sub_text = "Kepercayaan Sangat Baik" if is_confident else "Perlu Konfirmasi Ulang"

                    st.markdown(f"""
                    <div class="kpi-row">
                        <div class="kpi-tile">
                            <div class="kpi-tile-label">{svg_icon("leaf", size=13, color="#5F6D64")} DIAGNOSIS TERIDENTIFIKASI</div>
                            <div class="kpi-tile-value" style="font-size:1.25rem; color:var(--agri-forest-dark);">{top_class}</div>
                            <div class="kpi-tile-sub"><em>{disease_info['scientific_name']}</em></div>
                        </div>
                        <div class="kpi-tile">
                            <div class="kpi-tile-label">{svg_icon("activity", size=13, color="#5F6D64")} TINGKAT KEYAKINAN</div>
                            <div class="kpi-tile-value">{top_conf:.1f}%</div>
                            <div class="kpi-tile-sub">
                                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{conf_dot_color};"></span>
                                {conf_sub_text}
                            </div>
                        </div>
                        <div class="kpi-tile">
                            <div class="kpi-tile-label">{svg_icon("clock", size=13, color="#5F6D64")} WAKTU KOMPUTASI</div>
                            <div class="kpi-tile-value">{latency:.1f} <span style="font-size:0.85rem; font-weight:500;">ms</span></div>
                            <div class="kpi-tile-sub">Inferensi MobileNetV2</div>
                        </div>
                        <div class="kpi-tile">
                            <div class="kpi-tile-label">{svg_icon("shield", size=13, color="#5F6D64")} TINGKAT URGENSI</div>
                            <div class="kpi-tile-value" style="color:{severity_color};">{severity}</div>
                            <div class="kpi-tile-sub">Kategori: {category}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Distribusi Keyakinan Seluruh Kelas (Ranking 1 - 6):**")
                    for item in results["all_predictions"]:
                        rank = item["rank"]
                        c_name = item["class"]
                        pct = item["pct"]
                        is_top = (rank == 1)
                        
                        bar_style = "background: linear-gradient(90deg, #173628 0%, #2D5F45 100%);" if is_top else "background: #BAC7BE;"
                        font_weight = "bold" if is_top else "normal"
                        
                        st.markdown(f"""
                        <div class="prob-item">
                            <div class="prob-header">
                                <span class="prob-name" style="font-weight:{font_weight};">{rank}. {c_name}</span>
                                <span class="prob-pct" style="font-weight:{font_weight};">{pct:.2f}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: {pct}%; {bar_style}"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="empty-placeholder-box">
                    <div class="empty-icon-circle">{svg_icon("activity", size=24, color="#173628")}</div>
                    <div class="empty-title">Menunggu Masukan Citra</div>
                    <p class="empty-desc">
                        Hasil diagnosis, skor keyakinan, dan visualisasi Grad-CAM akan tampil di panel ini setelah gambar daun dipilih di sebelah kiri.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Grad-CAM Explainable AI & Agronomic Guide
    # ─────────────────────────────────────────
    if image_to_process is not None and is_leaf:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">3</span>
                <span>Visualisasi Analisis Atensi (Grad-CAM)</span>
            </div>
            <p style="color:#5F6D64; font-size:0.9rem; margin-bottom:1.2rem;">
                Grad-CAM (<em>Gradient-weighted Class Activation Mapping</em>) memetakan kontribusi fitur visual pada lapisan konvolusi terakhir (<code>features.18</code>).
                Regio bergradasi merah/kuning terang menunjukkan area daun yang paling memengaruhi keputusan model klasifikasi.
            </p>
            """, unsafe_allow_html=True)

            with st.spinner("Menghitung gradien aktivasi dan heatmap Grad-CAM..."):
                try:
                    cam_norm, _ = cam_extractor.generate(input_tensor.to(device), class_idx=pred_idx)
                    overlay_img, pure_heatmap_img = overlay_heatmap(
                        image_to_process, 
                        cam_norm, 
                        alpha=cam_alpha, 
                        colormap=cam_cmap
                    )
                except Exception as e:
                    st.error(f"Gagal menghitung Grad-CAM: {e}")
                    overlay_img = None
                    pure_heatmap_img = None

            if overlay_img is not None:
                col_v1, col_v2, col_v3 = st.columns(3, gap="medium")
                with col_v1:
                    st.markdown("<p style='text-align:center; font-weight:600; font-size:0.88rem; color:#1A241F;'>① Gambar Daun Asli</p>", unsafe_allow_html=True)
                    st.image(image_to_process, use_container_width=True)
                with col_v2:
                    st.markdown(f"<p style='text-align:center; font-weight:600; font-size:0.88rem; color:#1A241F;'>② Heatmap Grad-CAM ({cam_cmap})</p>", unsafe_allow_html=True)
                    st.image(pure_heatmap_img, use_container_width=True)
                with col_v3:
                    st.markdown(f"<p style='text-align:center; font-weight:600; font-size:0.88rem; color:#1A241F;'>③ Overlay Visualisasi (Alpha: {cam_alpha})</p>", unsafe_allow_html=True)
                    st.image(overlay_img, use_container_width=True)

        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">4</span>
                <span>Rekomendasi Solusi & Penanganan: {top_class} ({indo_name})</span>
            </div>
            <p style="font-size:0.95rem; line-height:1.6; color:#2C3E33; margin-bottom:1.4rem;">
                <strong>Ringkasan Kondisi:</strong> {disease_info['summary']}
            </p>
            """, unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2, gap="large")

            with col_g1:
                st.markdown(f"""
                <div class="agri-solution-box" style="border-left-color: #2D5F45;">
                    <div class="agri-solution-head">{svg_icon("microscope", size=16, color="#2D5F45")} Gejala Visual Utama</div>
                """, unsafe_allow_html=True)
                for symptom in disease_info["symptoms"]:
                    st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.35rem;'>• {symptom}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="agri-solution-box" style="border-left-color: #A34832;">
                    <div class="agri-solution-head">{svg_icon("alert-triangle", size=16, color="#A34832")} Faktor Penyebab / Patogen</div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.86rem; color:#405247;'>{disease_info['cause']}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="agri-solution-box" style="border-left-color: #173628;">
                    <div class="agri-solution-head">{svg_icon("shield", size=16, color="#173628")} Tindakan Preventif & Kultur Teknis</div>
                """, unsafe_allow_html=True)
                for prev in disease_info["prevention"]:
                    st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.35rem;'>• {prev}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_g2:
                st.markdown(f"""
                <div class="agri-solution-box" style="border-left-color: #554477;">
                    <div class="agri-solution-head">{svg_icon("flask", size=16, color="#554477")} Pengendalian Kimiawi (Sesuai Anjuran)</div>
                """, unsafe_allow_html=True)
                for chem in disease_info["chemical_control"]:
                    st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.35rem;'>• {chem}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="agri-solution-box" style="border-left-color: #2E7D32;">
                    <div class="agri-solution-head">{svg_icon("leaf", size=16, color="#2E7D32")} Pengendalian Organik & Ramah Lingkungan</div>
                """, unsafe_allow_html=True)
                for org in disease_info["organic_control"]:
                    st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.35rem;'>• {org}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            report_text = f"""=== LAPORAN HASIL DIAGNOSIS PENYAKIT DAUN CABAI ===
Sistem           : LeafCare — Chili Disease Detection
Model AI         : MobileNetV2 (Transfer Learning)
Validasi Citra   : Terverifikasi Daun Tanaman Cabai
Target Kelas     : {top_class} ({indo_name})
Skor Keyakinan   : {top_conf:.2f}%
Waktu Inferensi  : {latency:.2f} ms
Tingkat Risiko   : {severity}
Kategori Patogen : {category}
Nama Ilmiah      : {disease_info['scientific_name']}

RINGKASAN:
{disease_info['summary']}

FAKTOR PENYEBAB:
{disease_info['cause']}

GEJALA:
""" + "\n".join([f"- {s}" for s in disease_info['symptoms']]) + """

PENGENDALIAN KIMIAWI:
""" + "\n".join([f"- {c}" for c in disease_info['chemical_control']]) + """

PENGENDALIAN ORGANIK:
""" + "\n".join([f"- {o}" for o in disease_info['organic_control']]) + """

PENCEGAHAN:
""" + "\n".join([f"- {p}" for p in disease_info['prevention']]) + """

PERINGKAT KEYAKINAN LENGKAP:
""" + "\n".join([f"{item['rank']}. {item['class']}: {item['pct']:.2f}%" for item in results['all_predictions']]) + """
===================================================
Dibuat oleh LeafCare Plant Protection System
"""
            st.download_button(
                label="Unduh Ringkasan Diagnosis (.txt)",
                data=report_text,
                file_name=f"Diagnosis_{top_class.replace(' ', '_')}.txt",
                mime="text/plain"
            )

# ═════════════════════════════════════════════
# HALAMAN 2: ENSIKLOPEDIA PENYAKIT
# ═════════════════════════════════════════════
elif current_page == "Ensiklopedia Penyakit":
    st.markdown(f"""
    <div class="inner-page-header">
        <div class="inner-page-title">
            {svg_icon("book", size=28, color="#173628")}
            <span>Ensiklopedia Penyakit Daun Cabai</span>
        </div>
        <p class="inner-page-subtitle">
            Panduan agronomi komprehensif mengenai 6 kondisi daun cabai yang dapat dideteksi oleh sistem kami.
        </p>
    </div>
    """, unsafe_allow_html=True)

    catalog_tabs = st.tabs([k for k in DISEASE_DATABASE.keys()])

    for idx, (d_name, d_info) in enumerate(DISEASE_DATABASE.items()):
        with catalog_tabs[idx]:
            with st.container(border=True):
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                    <div>
                        <div style="font-family:'Playfair Display', serif; font-size:1.6rem; font-weight:700; color:var(--agri-text-dark); display:flex; align-items:center; gap:0.5rem;">
                            {svg_icon(d_info.get('icon_name', 'leaf'), size=24, color="#173628")}
                            <span>{d_name}</span>
                        </div>
                        <span style="font-size:1rem; color:var(--agri-forest-dark); font-weight:700;">{d_info['indonesian_name']}</span> • 
                        <em style="color:var(--agri-text-muted);">{d_info['scientific_name']}</em>
                    </div>
                    <span style="background:{d_info['severity_color']}; color:white; font-size:0.82rem; padding:0.4rem 0.95rem; border-radius:20px; font-weight:700;">
                        Tingkat Risiko: {d_info['severity']}
                    </span>
                </div>
                <p style="font-size:1rem; line-height:1.65; color:#36463E; margin-bottom:1.4rem;">
                    {d_info['summary']}
                </p>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.markdown(f"""
                    <div class="agri-solution-box" style="border-left-color: #2D5F45;">
                        <div class="agri-solution-head">{svg_icon("microscope", size=16, color="#2D5F45")} Gejala & Ciri Visual</div>
                    """, unsafe_allow_html=True)
                    for s in d_info["symptoms"]:
                        st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.35rem;'>• {s}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:0.8rem; font-size:0.86rem; color:#405247;'><strong>Penyebab:</strong> {d_info['cause']}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                    <div class="agri-solution-box" style="border-left-color: #173628;">
                        <div class="agri-solution-head">{svg_icon("shield", size=16, color="#173628")} Strategi Pengendalian & Solusi</div>
                    """, unsafe_allow_html=True)
                    st.markdown("<div style='font-size:0.86rem; font-weight:700; color:#1A241F; margin-bottom:0.3rem;'>Pengendalian Kimiawi:</div>", unsafe_allow_html=True)
                    for c in d_info["chemical_control"]:
                        st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.25rem;'>• {c}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='font-size:0.86rem; font-weight:700; color:#1A241F; margin:0.7rem 0 0.3rem 0;'>Pengendalian Organik & Hayati:</div>", unsafe_allow_html=True)
                    for o in d_info["organic_control"]:
                        st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.25rem;'>• {o}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='font-size:0.86rem; font-weight:700; color:#1A241F; margin:0.7rem 0 0.3rem 0;'>Pencegahan:</div>", unsafe_allow_html=True)
                    for p in d_info["prevention"]:
                        st.markdown(f"<div style='font-size:0.86rem; color:#405247; margin-bottom:0.25rem;'>• {p}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# HALAMAN 3: SPESIFIKASI MODEL
# ═════════════════════════════════════════════
elif current_page == "Spesifikasi Model":
    st.markdown(f"""
    <div class="inner-page-header">
        <div class="inner-page-title">
            {svg_icon("cpu", size=28, color="#173628")}
            <span>Arsitektur Model & Metodologi Pelatihan</span>
        </div>
        <p class="inner-page-subtitle">
            Detail teknis arsitektur MobileNetV2, guardrail validasi daun cabai, pra-pemrosesan tensor, dan pipeline model <code>mobilenetv2_chili.pth</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1, 1], gap="large")

    with col_t1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">A</span>
                <span>Karakteristik Arsitektur MobileNetV2</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            - **Arsitektur Dasar**: `{metadata.get('arch', 'mobilenet_v2')}`
            - **Total Parameter**: `{metadata.get('total_params', 2231558):,}` parameter
            - **Bobot Dapat Dilatih (Trainable)**: `{metadata.get('trainable_params', 2231558):,}` parameter
            - **Resolusi Masukan (Input Size)**: `{metadata.get('img_size', 224)} × {metadata.get('img_size', 224)} × 3 (RGB)`
            - **Dropout Classifier Layer**: `{metadata.get('dropout', 0.3)}`
            - **Jumlah Kelas Output**: `{len(class_names)} Kelas`
            - **Target Layer Grad-CAM**: `features[18]` (Inverted Residual Block Terakhir)
            - **Perangkat Komputasi**: `{device.upper()}`
            - **File Model Path**: `{metadata.get('model_path', 'models/mobilenetv2_chili.pth')}`
            """)

        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">B</span>
                <span>Guardrail Validasi Citra (Dual-Stage OOD Detector)</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            Mekanisme pencegahan tebakan acak pada gambar bukan daun cabai:
            - **Tahap 1 (Botanical CV Filter)**:
              - *HSV Foliage Masking*: Memindai klorofil hijau, klorosis kuning, dan nekrosis cokelat.
              - *Venation Micro-Texture Density*: Memverifikasi kerapatan pola pertulangan daun alami untuk menolak objek hijau halus (botol, apel, bola, plastik).
              - *Natural Pigment Dispersion*: Memastikan variasi rona (*hue std*) khas klorofil biologis untuk menolak warna hijau sintetis.
              - *Alien Color Rejection*: Menolak dominasi warna biru/ungu non-tanaman.
            - **Tahap 2 (Deep Energy & Logit Domain Filter)**:
              - *Maximum Logit Gate*: Mengukur aktivasi neuron kelas tertinggi pada MobileNetV2 (min. 6.5) untuk memastikan daun berada dalam domain tanaman cabai.
              - *Free Energy Score*: Mendeteksi dan menolak daun tanaman lain (*Out-of-Distribution*) sebelum probabilitas softmax memaksakan tebakan acak.
              - *Confidence Gate*: Memperingatkan pengguna apabila keyakinan inferensi di bawah 55%.
            """)

    with col_t2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">C</span>
                <span>Pemetaan Indeks Kelas (Class Mapping)</span>
            </div>
            """, unsafe_allow_html=True)
            
            mapping_data = []
            for i, name in enumerate(class_names):
                d_item = get_disease_details(name)
                mapping_data.append({
                    "Index": i,
                    "Nama Kelas (Model)": name,
                    "Nama Bahasa Indonesia": d_item["indonesian_name"],
                    "Kategori": d_item["category"]
                })
            
            df_mapping = pd.DataFrame(mapping_data)
            st.dataframe(df_mapping, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">D</span>
                <span>Prinsip Kerja Grad-CAM</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            1. **Forward Pass**: Gambar input dialirkan melalui seluruh lapisan konvolusi MobileNetV2 hingga menghasilkan skor aktivasi logit.
            2. **Backward Pass**: Gradien dari skor kelas terprediksi dihitung mundur terhadap peta fitur aktivasi di lapisan konvolusi terakhir.
            3. **Global Average Pooling**: Gradien dirata-ratakan secara spasial untuk mendapatkan bobot pentingnya masing-masing kanal fitur.
            4. **Linear Combination & ReLU**: Menggabungkan peta fitur berdasarkan bobot positifnya untuk menyaring fitur-fitur pendukung keputusan diagnosis.
            5. **Color Overlay**: Heatmap dinormalisasi dan diinterpolasi ke resolusi gambar asli menggunakan colormap OpenCV (JET / TURBO / VIRIDIS).
            """)

# ═════════════════════════════════════════════
# HALAMAN 4: TENTANG KAMI
# ═════════════════════════════════════════════
elif current_page == "Tentang Kami":
    st.markdown(f"""
    <div class="inner-page-header">
        <div class="inner-page-title">
            {svg_icon("sprout", size=28, color="#173628")}
            <span>Tentang LeafCare</span>
        </div>
        <p class="inner-page-subtitle">
            Mewujudkan Pertanian Presisi untuk Mendukung Produktivitas Petani Cabai Indonesia.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_ab1, col_ab2 = st.columns([1.2, 1], gap="large")

    with col_ab1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">1</span>
                <span>Visi & Misi Inovasi</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **LeafCare** dikembangkan sebagai instrumen digital pendukung petani cabai (*Capsicum annuum L.*) dalam melakukan identifikasi dini dan penanganan penyakit daun secara akurat.
            
            - **Presisi Berbasis Data**: Memanfaatkan model Convolutional Neural Network (CNN) MobileNetV2 yang dioptimalkan dengan Transfer Learning dan mencapai akurasi uji sebesar **94.22%**.
            - **Explainable AI (XAI)**: Menyediakan visualisasi peta panas Grad-CAM sehingga petani dan penyuluh pertanian dapat melihat bagian daun mana yang mendasari analisis model.
            - **Keamanan Diagnostik (OOD Guardrail)**: Mencegah terjadinya salah tebak saat sistem menerima gambar yang bukan daun cabai.
            - **Solusi Agronomi Komprehensif**: Mengintegrasikan rekomendasi pengendalian kimiawi, hayati/organik, serta tindakan pencegahan preventif.
            """)

    with col_ab2:
        with st.container(border=True):
            st.markdown(f"""
            <div class="studio-section-head">
                <span class="step-badge">2</span>
                <span>Informasi Proyek & Lingkungan Sistem</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            - **Sistem**: `LeafCare Plant Protection AI`
            - **Model Backend**: `PyTorch MobileNetV2 (0.3 Dropout)`
            - **Perangkat Aktif**: `{device.upper()}`
            - **Dataset Pelatihan**: 6 Kelas Penyakit Daun Cabai
            - **Resolusi Pemrosesan**: `224 × 224 px (ImageNet Normalized)`
            - **Framework Antarmuka**: `Streamlit 1.66`
            - **Tahun Rilis**: `2026`
            """)
            
            st.markdown("""
            <div style="background:#F1F6F2; border:1px solid #D1E2D7; border-radius:12px; padding:1rem; margin-top:1rem; text-align:center;">
                <span style="font-weight:700; color:var(--agri-forest-dark);">Dukung Petani Indonesia Menuju Pertanian Cerdas</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BOTTOM BANNER (Matching LeafCare Reference)
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="agri-bottom-banner">
    <div class="bottom-brand-group">
        <div class="bottom-icon-box">{svg_icon("leaf", size=22, color="#173628")}</div>
        <div class="bottom-tagline">
            Together, let's build a greener and more prosperous tomorrow.
        </div>
    </div>
    <div class="bottom-pills">
        <div class="bottom-pill-item">
            {svg_icon("sprout", size=15, color="#173628")}
            <span>Sustainable Farming</span>
        </div>
        <div class="bottom-pill-item">
            {svg_icon("chart", size=15, color="#173628")}
            <span>Better Yield</span>
        </div>
        <div class="bottom-pill-item">
            {svg_icon("shield", size=15, color="#173628")}
            <span>Stronger Communities</span>
        </div>
        <a href="?page=Studio+Deteksi#studio-section" target="_self" style="background:var(--agri-forest-dark); color:white; padding:0.5rem 1.2rem; border-radius:25px; font-size:0.82rem; font-weight:600; text-decoration:none;">
            LeafCare
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
