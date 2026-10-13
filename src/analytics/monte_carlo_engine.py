import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def run_monte_carlo_emissions(
    df: pd.DataFrame, 
    n_simulations: int = 10000
) -> dict:
    """
    Simulasi Monte Carlo (10.000 iterasi) untuk mengestimasi ketidakpastian
    total emisi karbon (ton CO2e/tahun) dari situs pembakaran sampah terbuka di Tangsel.
    """
    if df.empty:
        logger.warning("DataFrame kosong, simulasi Monte Carlo dibatalkan.")
        return {}

    # Hanya hitung lokasi yang teridentifikasi memiliki aktivitas pembakaran terbuka
    active_sites = df[df.get("open_burning_flag", 1) == 1]
    n_sites = len(active_sites)

    if n_sites == 0:
        logger.warning("Tidak ditemukan situs pembakaran aktif untuk simulasi emisi.")
        return {}

    np.random.seed(42)

    # Parameter Distribusi Probabilitas Emisi
    # Mass waste burned per site (ton/hari) ~ Triangular Distribution (min=0.2, mode=0.5, max=1.5)
    waste_burned_daily = np.random.triangular(0.2, 0.5, 1.5, size=(n_simulations, n_sites))
    
    # Emission Factor (kg CO2e / ton sampah terbakar) ~ Normal Distribution (mean=1100, std=150)
    emission_factor = np.random.normal(1100, 150, size=(n_simulations, n_sites))

    # Estimasi Total Emisi Tahunan (ton CO2e / tahun)
    daily_emissions_kg = np.sum(waste_burned_daily * emission_factor, axis=1)
    annual_emissions_ton = (daily_emissions_kg * 365) / 1000.0

    # Kalkulasi Interval Kepercayaan (Confidence Interval 95%)
    ci_lower = np.percentile(annual_emissions_ton, 2.5)
    mean_emissions = np.mean(annual_emissions_ton)
    ci_upper = np.percentile(annual_emissions_ton, 97.5)

    results = {
        "active_sites_count": int(n_sites),
        "mean_annual_co2e_tons": float(np.round(mean_emissions, 2)),
        "ci_95_lower": float(np.round(ci_lower, 2)),
        "ci_95_upper": float(np.round(ci_upper, 2)),
        "simulations_run": n_simulations
    }

    logger.info(f"Monte Carlo Simulation Complete ({n_simulations} iterations). Mean CO2e: {mean_emissions:.2f} tons/year.")
    return results