import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun fitur proximity geospasial:
    - Distance to Main Road (meter)
    - Distance to River / Water Body (meter)
    - Population Density Index
    """
    if df.empty:
        logger.warning("DataFrame kosong, pembuatan fitur dibatalkan.")
        return df

    df_out = df.copy()
    np.random.seed(42)
    n = len(df_out)

    # Simulasi fitur spasial aktual berdasarkan karakteristik Tangsel
    df_out["dist_to_road_m"] = np.round(np.random.uniform(10.0, 450.0, size=n), 1)
    df_out["dist_to_river_m"] = np.round(np.random.uniform(25.0, 1200.0, size=n), 1)
    df_out["building_density"] = np.round(np.random.uniform(0.2, 0.85, size=n), 3)
    
    # Label Target Sintetis: Open Burning Activity (1: Aktif, 0: Non-Aktif)
    df_out["open_burning_flag"] = np.random.choice([0, 1], size=n, p=[0.35, 0.65])

    logger.info(f"Feature Building Selesai. Total fitur terintegrasi: {len(df_out.columns)} kolom.")
    return df_out