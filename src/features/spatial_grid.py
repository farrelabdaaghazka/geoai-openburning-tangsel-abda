import logging
from typing import List
import pandas as pd
import h3

logger = logging.getLogger(__name__)

def assign_h3_grid(df: pd.DataFrame, resolution: int, col_name: str = "h3_index") -> pd.DataFrame:
    """
    Menambahkan kolom Uber H3 Index berdasarkan koordinat latitude dan longitude.
    Kompatibel dengan h3-py v3.x dan v4.x.
    """
    if df.empty:
        logger.warning("DataFrame kosong, mengembalikan DataFrame asal.")
        return df

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise KeyError("DataFrame harus memiliki kolom 'latitude' dan 'longitude'.")

    df_out = df.copy()

    # Dynamic fallback untuk mendukung h3-py versi 3.x dan 4.x
    def get_h3_cell(lat, lon, res):
        if hasattr(h3, 'latlng_to_cell'):
            return h3.latlng_to_cell(lat, lon, res)
        elif hasattr(h3, 'geo_to_h3'):
            return h3.geo_to_h3(lat, lon, res)
        elif hasattr(h3, 'latlng_to_h3'):
            return h3.latlng_to_h3(lat, lon, res)
        else:
            raise AttributeError("Modul H3 tidak memiliki fungsi pemetaan koordinat yang sesuai.")

    df_out[col_name] = df_out.apply(
        lambda row: get_h3_cell(row["latitude"], row["longitude"], resolution),
        axis=1
    )

    logger.info(f"Berhasil memetakan {len(df_out)} baris data ke H3 Grid Resolusi {resolution}.")
    return df_out


def get_h3_neighbors(h3_index: str, ring_radius: int = 1) -> List[str]:
    """
    Mendapatkan daftar indeks H3 tetangga di sekitar sel acuan.
    """
    try:
        if hasattr(h3, 'grid_disk'):
            return list(h3.grid_disk(h3_index, ring_radius))
        elif hasattr(h3, 'k_ring'):
            return list(h3.k_ring(h3_index, ring_radius))
    except Exception as err:
        logger.error(f"Gagal mendapatkan tetangga H3 untuk index '{h3_index}': {err}")
    return []


def generate_full_tangsel_h3_grid(min_lat: float = -6.38, max_lat: float = -6.25, 
                                  min_lon: float = 106.63, max_lon: float = 106.78, 
                                  resolution: int = 8) -> pd.DataFrame:
    """
    Membangkitkan seluruh grid heksagon H3 yang menutupi batas wilayah Tangerang Selatan.
    Mengatasi masalah data yang terlalu sedikit (data sparsity) dengan menghasilkan
    ruang analisis spasial yang utuh (full coverage).
    """
    boundary_polygon = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon)
    ]
    
    # Menghasilkan sel H3 penutup
    try:
        if hasattr(h3, 'polygon_to_cells'):
            # h3 v4.x
            geojson_poly = {
                "type": "Polygon",
                "coordinates": [[[lon, lat] for lat, lon in boundary_polygon + [boundary_polygon[0]]]]
            }
            hexagons = list(h3.polygon_to_cells(geojson_poly, resolution))
        elif hasattr(h3, 'polyfill'):
            # h3 v3.x
            geojson_poly = {
                "type": "Polygon",
                "coordinates": [[[lon, lat] for lat, lon in boundary_polygon + [boundary_polygon[0]]]]
            }
            hexagons = list(h3.polyfill(geojson_poly, resolution, geo_json=True))
        else:
            logger.warning("Fungsi polyfill H3 tidak ditemukan, menggunakan titik centroid fallback.")
            hexagons = []
    except Exception as e:
        logger.error(f"Gagal membangkitkan grid H3 Tangsel: {e}")
        hexagons = []

    records = []
    for h3_id in hexagons:
        if hasattr(h3, 'cell_to_latlng'):
            lat, lon = h3.cell_to_latlng(h3_id)
        else:
            lat, lon = h3.h3_to_geo(h3_id)
            
        records.append({
            "h3_index": h3_id,
            "latitude": lat,
            "longitude": lon,
            "location_name": f"H3 Grid Cell {h3_id[:8]}"
        })

    df_grid = pd.DataFrame(records)
    logger.info(f"Berhasil membangkitkan {len(df_grid)} sel H3 Grid penutup wilayah Tangsel (Resolusi {resolution}).")
    return df_grid