import asyncio
import logging
import pandas as pd

from config.spatial_config import SpatialConfig
from src.ingestion.gmaps_scraper import scrape_gmaps_poi
from src.ingestion.text_geocoder import geocode_realtime_keywords
from src.ingestion.sentinel_extractor import extract_sentinel_indices
from src.features.spatial_grid import assign_h3_grid
from src.features.feature_builder import build_spatial_features
from src.models.spatial_split import apply_spatial_block_split
from src.models.causal_dml import estimate_causal_effect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GeoAIPipelineRunner")

async def run_pipeline():
    logger.info("Starting GeoAI Ingestion & Spatial Partitioning Pipeline...")
    config = SpatialConfig()

    # 1. Ingestion Data POI & Text Geocoding (Data Riil Otomatis)
    logger.info("Phase 1A: Scraping Google Maps POIs...")
    df_poi = await scrape_gmaps_poi(config.poi_queries, config)

    logger.info("Phase 1B: Geocoding Real-time Tangsel Landmarks...")
    real_landmarks = [
        "Pasar Ciputat", 
        "Lapangan Sunburst Serpong", 
        "UIN Syarif Hidayatullah", 
        "Alun Alun Pamulang",
        "Taman Kota 1 BSD"
    ]
    df_text = geocode_realtime_keywords(real_landmarks, config)

    df_combined = pd.concat([df_poi, df_text], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)

    # 2. Sentinel-2 & Spatial Feature Extraction
    logger.info("Phase 2A: Extracting Sentinel-2 Spectral Indices (NDVI, NDBI, BSI)...")
    df_sentinel = extract_sentinel_indices(df_combined)

    logger.info("Phase 2B: Building Spatial Proximity Features...")
    df_featured = build_spatial_features(df_sentinel)

    # 3. Uber H3 Gridding
    logger.info("Phase 3: Mapping Coordinates to Uber H3 Grid...")
    df_processed = (
        df_featured
        .pipe(assign_h3_grid, resolution=config.h3_resolution_micro, col_name="h3_grid_micro")
        .pipe(assign_h3_grid, resolution=config.h3_resolution_macro, col_name="h3_block_macro")
    )

    # 4. Spatial Block Partitioning (GroupKFold)
    logger.info("Phase 4: Partitioning Dataset with Spatial Block Split...")
    df_final = apply_spatial_block_split(df_processed, "h3_block_macro", n_splits=5)

    # 5. Causal Machine Learning (Double DML)
    logger.info("Phase 5: Estimating Causal Impact with Double Machine Learning (DML)...")
    causal_results = estimate_causal_effect(
        df_final[df_final["split_role"] == "TRAIN"],
        treatment_col="bsi",
        outcome_col="open_burning_flag"
    )

    # Save Output Data
    output_path = "data/processed/tangsel_spatial_partitioned.csv"
    df_final.to_csv(output_path, index=False)
    logger.info(f"Pipeline Execution Complete. Saved output to '{output_path}'.")

    print("\n--- CAUSAL DML MODEL RESULTS ---")
    print(causal_results)

if __name__ == "__main__":
    asyncio.run(run_pipeline())