import sys
from pathlib import Path

# Registrasi root directory ke sys.path agar tidak terjadi ModuleNotFoundError saat impor 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from src.visualization.generate_interactive_map import create_interactive_web_map

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Spatial Risk Analysis - Open Burning Tangsel",
    layout="wide"
)

# 2. Custom CSS untuk Metric Cards
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# Native Header (Hanya Judul Utama Tanpa Subtitle Caption)
st.title("GeoAI Open Burning Risk & Waste Infrastructure WebGIS")

data_path = "data/processed/tangsel_spatial_partitioned.csv"

if not os.path.exists(data_path):
    st.error(f"File data `{data_path}` belum ditemukan. Jalankan `python pipeline_runner.py` terlebih dahulu.")
else:
    # Memperbarui HTML Peta Interaktif
    html_map_path = create_interactive_web_map(data_path)
    df = pd.read_csv(data_path)

    # Layout Split: 75% WebGIS Map | 25% Analytics Panel
    col_map, col_analytics = st.columns([3.4, 1.2])

    with col_map:
        if html_map_path and os.path.exists(html_map_path):
            with open(html_map_path, "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=660, scrolling=False)

    with col_analytics:
        st.subheader("Analytics Summary")
        
        # Perhitungan Ringkasan Statistik
        total_sites = len(df)
        high_risk_sites = len(df[df["open_burning_flag"] == 1]) if "open_burning_flag" in df.columns else 0
        rec_facilities = len(df[df["recommended_facility_site"] == True]) if "recommended_facility_site" in df.columns else 0
        avg_bsi = df["bsi"].mean() if "bsi" in df.columns else 0.0

        # Custom Metric Cards Grid 2x2
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Sites</div>
                    <div class='metric-value'>{total_sites}</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Hotspots</div>
                    <div class='metric-value' style='color:#e74c3c;'>{high_risk_sites}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>TPS3R Rec</div>
                    <div class='metric-value' style='color:#2ecc71;'>{rec_facilities}</div>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Mean BSI</div>
                    <div class='metric-value'>{avg_bsi:.3f}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Histogram BSI
        if "bsi" in df.columns:
            st.markdown("**Distribusi Indeks Keterbukaan Lahan (BSI)**")
            fig = px.histogram(
                df, 
                x="bsi", 
                nbins=12, 
                color_discrete_sequence=['#34495e'],
                height=180
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Bare Soil Index (BSI)",
                yaxis_title="Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.download_button(
            label="Export Dataset (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="tangsel_open_burning_processed.csv",
            mime="text/csv",
            use_container_width=True
        )