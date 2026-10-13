import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

def estimate_causal_effect(
    df: pd.DataFrame,
    treatment_col: str = "bsi",
    outcome_col: str = "open_burning_flag",
    confounder_cols: list = None
) -> dict:
    """
    Mengestimasi Average Treatment Effect (ATE) dan Standard Error (SE) menggunakan
    Double Machine Learning (DML) dengan Cross-Fitting ortogonal untuk mengisolasi
    efek kausalitas murni dari bias confounding.
    """
    if df.empty:
        logger.warning("DataFrame kosong, estimasi Causal DML dibatalkan.")
        return {}

    if confounder_cols is None:
        # Menyesuaikan dengan variabel kontrol spasial & lingkungan built environment
        confounder_cols = ["ndvi", "ndbi", "dist_to_road_m", "dist_to_river_m"]

    # Filter hanya kolom confounders yang tersedia di DataFrame
    available_confounders = [c for c in confounder_cols if c in df.columns]
    
    # Memastikan kolom treatment dan outcome tersedia
    missing_cols = [c for c in [treatment_col, outcome_col] if c not in df.columns]
    if missing_cols or not available_confounders:
        logger.error(f"Kolom tidak lengkap untuk Causal DML. Missing: {missing_cols}")
        raise KeyError(f"Kolom tidak ditemukan untuk DML: {missing_cols}")

    X = df[available_confounders].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    n_samples = len(df)

    # Penyesuaian K-Fold dinamis agar tidak crash jika data kecil
    n_splits = max(2, min(5, n_samples // 2)) if n_samples >= 4 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    T_res = np.zeros_like(T, dtype=float)
    Y_res = np.zeros_like(Y, dtype=float)

    # Step 1: Orthogonalization via Cross-Fitting
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Model Nuisance 1: Prediksi Treatment (BSI) berdasarkan Confounders
        model_t = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_t.fit(X_train, T[train_idx])
        T_res[test_idx] = T[test_idx] - model_t.predict(X_test)

        # Model Nuisance 2: Prediksi Outcome (Open Burning Flag) berdasarkan Confounders
        # Memastikan penanganan jika class di fold training seragam (single class)
        if len(np.unique(Y[train_idx])) > 1:
            model_y = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model_y.fit(X_train, Y[train_idx])
            Y_res[test_idx] = Y[test_idx] - model_y.predict_proba(X_test)[:, 1]
        else:
            Y_res[test_idx] = Y[test_idx] - np.mean(Y[train_idx])

    # Step 2: Causal Regression pada Residual
    # ATE = Cov(Y_res, T_res) / Var(T_res)
    denom = np.sum(T_res ** 2)
    if denom != 0:
        ate = float(np.sum(T_res * Y_res) / denom)
        # Menghitung Standard Error (SE) & p-value untuk kebutuhan rincian di paper
        residuals = Y_res - (ate * T_res)
        se = float(np.sqrt(np.sum(residuals ** 2) / ((n_samples - 1) * denom)))
        t_stat = ate / se if se > 0 else 0.0
        p_val = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
    else:
        ate, se, p_val = 0.0, 0.0, 1.0

    results = {
        "ate_estimate": ate,
        "standard_error": se,
        "p_value": p_val,
        "treatment_variable": treatment_col,
        "outcome_variable": outcome_col,
        "confounders_used": available_confounders,
        "n_samples": n_samples
    }

    logger.info(f"Causal DML Estimation Complete. ATE ({treatment_col} -> {outcome_col}): {ate:.4f} (SE: {se:.4f}, p: {p_val:.4f})")
    return results