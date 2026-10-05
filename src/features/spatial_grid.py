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