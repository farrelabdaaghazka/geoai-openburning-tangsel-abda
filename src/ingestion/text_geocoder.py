import logging
from typing import List, Dict, Any
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from config.spatial_config import SpatialConfig

logger = logging.getLogger(__name__)

def geocode_realtime_keywords(landmarks: List[str], config: SpatialConfig) -> pd.DataFrame:
    """
    Mengonversi daftar landmark / titik rawan aduan riil di Tangsel 
    menjadi koordinat spasial menggunakan OpenStreetMap Nominatim.
    """
    if not landmarks:
        return pd.DataFrame()

    geolocator = Nominatim(user_agent="geoai_tangsel_research_app_v1", timeout=10)
    rate_limited_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5, error_wait_seconds=3.0)

    viewbox_coords = [(config.min_lat, config.min_lon), (config.max_lat, config.max_lon)]
    results: List[Dict[str, Any]] = []

    for landmark in landmarks:
        query_string = f"{landmark}, Tangerang Selatan"
        try:
            location = rate_limited_geocode(
                query=query_string,
                viewbox=viewbox_coords,
                bounded=True
            )
            if location and (config.min_lon <= location.longitude <= config.max_lon) and (config.min_lat <= location.latitude <= config.max_lat):
                results.append({
                    "source": "OpenStreetMap_Geocoded_Landmark",
                    "location_name": landmark,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "confidence_score": 0.80
                })
        except Exception as err:
            logger.error(f"Geocoding error untuk '{landmark}': {err}")

    df_out = pd.DataFrame(results)
    logger.info(f"Geocoding landmark riil tuntas: {len(df_out)} lokasi berhasil dipetakan.")
    return df_out