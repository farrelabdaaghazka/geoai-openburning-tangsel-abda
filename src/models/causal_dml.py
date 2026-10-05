import logging
import numpy as np
import pandas as pd
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
    Mengestimasi Average Treatment Effect (ATE) menggunakan Double Machine Learning (DML)
    dengan Cross-Fitting untuk mengisolasi efek kausalitas murni dari bias confounding.
    """
    if df.empty:
        logger.warning("DataFrame kosong, estimasi Causal DML dibatalkan.")
        return {}

    if confounder_cols is None:
        confounder_cols = ["ndvi", "ndbi", "dist_to_road_m", "dist_to_river_m", "building_density"]

    # Memastikan kolom yang dibutuhkan tersedia
    missing_cols = [c for c in [treatment_col, outcome_col] + confounder_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Kolom tidak ditemukan untuk DML: {missing_cols}")

    X = df[confounder_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    T_res = np.zeros_like(T, dtype=float)
    Y_res = np.zeros_like(Y, dtype=float)

    # Step 1: Orthogonalization via Cross-Fitting
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Model Nuisance 1: Prediksi Treatment berdasarkan Confounders
        model_t = RandomForestRegressor(n_estimators=50, random_state=42)
        model_t.fit(X_train, T[train_idx])
        T_res[test_idx] = T[test_idx] - model_t.predict(X_test)

        # Model Nuisance 2: Prediksi Outcome berdasarkan Confounders
        model_y = RandomForestClassifier(n_estimators=50, random_state=42)
        model_y.fit(X_train, Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - model_y.predict_proba(X_test)[:, 1]

    # Step 2: Causal Regression pada Residual
    # ATE = Cov(Y_res, T_res) / Var(T_res)
    ate = np.sum(T_res * Y_res) / np.sum(T_res ** 2) if np.sum(T_res ** 2) != 0 else 0.0

    results = {
        "ate_estimate": float(ate),
        "treatment_variable": treatment_col,
        "outcome_variable": outcome_col,
        "n_samples": len(df)
    }

    logger.info(f"Causal DML Estimation Complete. ATE ({treatment_col} -> {outcome_col}): {ate:.4f}")
    return results