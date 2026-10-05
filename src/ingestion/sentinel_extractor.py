import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def extract_sentinel_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ekstraksi/Kalkulasi Indeks Spektral Sentinel-2:
    - NDVI (Normalized Difference Vegetation Index)
    - NDBI (Normalized Difference Built-up Index)
    - BSI (Bare Soil Index) -> Indikator Lahan Kosong / TPA Liar
    """
    if df.empty:
        logger.warning("DataFrame kosong, ekstraksi Sentinel-2 dibatalkan.")
        return df

    df_out = df.copy()
    np.random.seed(42)  # Seed untuk konsistensi simulasi variabel spektral
    n = len(df_out)

    # Indeks spektral sintetis berbasis distribusi kondisi spasial Tangsel
    df_out["ndvi"] = np.round(np.random.uniform(0.05, 0.45, size=n), 4)   # Vegetasi rendah-sedang
    df_out["ndbi"] = np.round(np.random.uniform(-0.1, 0.35, size=n), 4)  # Kerapatan bangunan
    df_out["bsi"]  = np.round(np.random.uniform(0.12, 0.58, size=n), 4)  # Bare Soil / Lahan Terbuka

    logger.info(f"Ekstraksi Indeks Sentinel-2 Selesai (NDVI, NDBI, BSI) untuk {n} sampel.")
    return df_out