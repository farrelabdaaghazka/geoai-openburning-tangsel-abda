import logging
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

def apply_spatial_block_split(
    df: pd.DataFrame, 
    spatial_group_col: str = "h3_block_macro", 
    n_splits: int = 5,
    test_fold: int = 0
) -> pd.DataFrame:
    """
    Membagi dataset secara spasial berdasarkan kluster H3 Macro Block
    guna mencegah kontaminasi autokorelasi spasial antara Train dan Test set.
    """
    if df.empty:
        logger.warning("DataFrame kosong, proses pemisahan dibatalkan.")
        return df

    if spatial_group_col not in df.columns:
        raise KeyError(f"Kolom kelompok spasial '{spatial_group_col}' tidak ditemukan dalam DataFrame.")

    df_out = df.copy()
    gkf = GroupKFold(n_splits=n_splits)
    df_out["spatial_fold"] = -1

    # Menggunakan koordinat sebagai matriks fitur X dummy untuk GroupKFold
    X = df_out[["latitude", "longitude"]]
    y = df_out.get("confidence_score", [1] * len(df_out))
    groups = df_out[spatial_group_col]

    for fold, (_, test_idx) in enumerate(gkf.split(X, y, groups)):
        df_out.loc[test_idx, "spatial_fold"] = fold

    # Menandai entri yang masuk ke dalam Test Set
    df_out["split_role"] = df_out["spatial_fold"].apply(
        lambda f: "TEST" if f == test_fold else "TRAIN"
    )

    train_count = len(df_out[df_out["split_role"] == "TRAIN"])
    test_count = len(df_out[df_out["split_role"] == "TEST"])
    logger.info(f"Spatial Partitioning Selesai: {train_count} Train | {test_count} Test (K-Fold={n_splits}).")

    return df_out