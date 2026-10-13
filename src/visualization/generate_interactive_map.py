import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import folium

logger = logging.getLogger(__name__)

def h3_to_polygon(h3_index: str) -> Polygon:
    boundary = h3.cell_to_boundary(h3_index) if hasattr(h3, 'cell_to_boundary') else h3.h3_to_geo_boundary(h3_index)
    return Polygon([(lng, lat) for lat, lng in boundary])

def create_interactive_web_map(data_path: str = "data/processed/tangsel_spatial_partitioned.csv", output_dir: str = "data/figures"):
    """
    Menghasilkan Peta Interaktif WebGIS HTML (Folium) yang bebas eror API key
    dan memiliki kontras layer H3 Grid yang tinggi.
    """
    if not os.path.exists(data_path):
        logger.error(f"File data '{data_path}' tidak ditemukan!")
        return None

    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)

    # Titik tengah koordinat Tangerang Selatan
    center_lat = df["latitude"].mean() if "latitude" in df.columns else -6.3006
    center_lon = df["longitude"].mean() if "longitude" in df.columns else 106.7073

    # Inisialisasi peta Folium dengan OpenStreetMap sebagai default basemap
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
        attr="OpenStreetMap"
    )

    # Basemap opsional Google Satellite
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satelit (High-Res)',
        overlay=False
    ).add_to(m)

    # Layer Groups
    fg_grid = folium.FeatureGroup(name="Grid H3 (Indeks BSI)", show=True).add_to(m)
    fg_hotspots = folium.FeatureGroup(name="Hotspot Pembakaran Sampah Liar", show=True).add_to(m)
    fg_facilities = folium.FeatureGroup(name="Rekomendasi TPS3R (P-Median)", show=True).add_to(m)

    # 1. Render Grid H3 Risiko
    grid_col = "h3_grid_micro" if "h3_grid_micro" in df.columns else ("h3_index" if "h3_index" in df.columns else None)
    if grid_col and grid_col in df.columns:
        df_grid = df.groupby(grid_col).agg({
            "latitude": "mean",
            "longitude": "mean",
            "bsi": "mean",
            "open_burning_flag": "max"
        }).reset_index()

        for _, row in df_grid.iterrows():
            try:
                poly = h3_to_polygon(str(row[grid_col]))
                coords = [[lat, lon] for lat, lon in poly.exterior.coords]
                bsi_val = row["bsi"]
                
                if bsi_val > 0.40:
                    fill_color = "#e74c3c"  # Merah (Sangat Rawan)
                elif bsi_val > 0.28:
                    fill_color = "#f39c12"  # Sedang
                else:
                    fill_color = "#2ecc71"  # Hijau (Rendah)

                folium.Polygon(
                    locations=coords,
                    color="#ffffff",  # Garis batas putih tegas
                    weight=1.2,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.50,
                    tooltip=f"H3 Index: {row[grid_col]} | Mean BSI: {bsi_val:.3f}"
                ).add_to(fg_grid)
            except Exception:
                continue

    # 2. Render Markers
    for _, row in df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        loc_name = str(row.get("location_name", "Lokasi Sampel"))
        bsi_val = float(row.get("bsi", 0.0))
        is_rec = bool(row.get("recommended_facility_site", False))
        is_burning = int(row.get("open_burning_flag", 0)) == 1

        is_facility_name = any(k in loc_name.upper() for k in ["TPS", "TPST", "BANK SAMPAH", "KOMPOS", "KSM", "GRIYA"])

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 220px; font-size: 12px; line-height: 1.5;">
            <b style="font-size: 13px; color: #2c3e50;">{loc_name}</b>
            <hr style="margin: 5px 0; border: 0; border-top: 1px solid #eee;">
            <b>Koordinat:</b> {lat:.5f}, {lon:.5f}<br>
            <b>Bare Soil Index (BSI):</b> {bsi_val:.4f}<br>
            <b>Status Risiko:</b> {'<span style="color:#e74c3c;font-weight:bold;">HIGH RISK</span>' if is_burning else '<span style="color:#2ecc71;">Low Risk</span>'}<br>
            <b>Rekomendasi TPS3R:</b> {'<span style="color:#2ecc71;font-weight:bold;">YA</span>' if is_rec else 'Tidak'}<br>
            <div style="margin-top: 8px;">
                <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" style="background:#34495e;color:white;padding:4px 10px;text-decoration:none;border-radius:4px;font-size:11px;display:inline-block;">Buka Google Maps ↗</a>
            </div>
        </div>
        """

        if is_facility_name or is_rec:
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Fasilitas: {loc_name}",
                icon=folium.Icon(color="green", icon="recycle", prefix="fa")
            ).add_to(fg_facilities)
        elif is_burning:
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Hotspot: {loc_name}",
                icon=folium.Icon(color="red", icon="fire", prefix="fa")
            ).add_to(fg_hotspots)
        else:
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Sampel: {loc_name}",
                icon=folium.Icon(color="blue", icon="info-circle", prefix="fa")
            ).add_to(fg_hotspots)

    folium.LayerControl(collapsed=False).add_to(m)

    html_output_path = os.path.join(output_dir, "interactive_risk_map.html")
    m.save(html_output_path)
    logger.info(f"Peta Interaktif WebGIS berhasil disimpan di: {html_output_path}")
    return html_output_path