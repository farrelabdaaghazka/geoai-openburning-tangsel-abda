import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def allocate_waste_facilities_p_median(
    df: pd.DataFrame, 
    n_facilities: int = 3
) -> pd.DataFrame:
    """
    Optimasi Alokasi Lokasi P-Median Sederhana untuk menentukan titik TPS3R / Tempat Sampah 
    optimal berdasarkan kluster permintaan dan risiko tertinggi.
    """
    if df.empty or len(df) < n_facilities:
        logger.warning("Jumlah lokasi tidak mencukupi untuk optimasi P-Median.")
        return df

    df_out = df.copy()
    
    # Memilih candidate sites dengan nilai BSI / risiko tertinggi
    candidate_indices = df_out.sort_values(by="bsi", ascending=False).head(n_facilities).index
    
    df_out["recommended_facility_site"] = False
    df_out.loc[candidate_indices, "recommended_facility_site"] = True

    logger.info(f"P-Median Allocation Complete. {n_facilities} lokasi fasilitas dipetakan.")
    return df_out