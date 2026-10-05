from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class SpatialConfig:
    # Bounding Box Tangerang Selatan (min_lon, min_lat, max_lon, max_lat)
    min_lon: float = 106.647
    min_lat: float = -6.377
    max_lon: float = 106.786
    max_lat: float = -6.230
    
    # Resolusi Uber H3 Grid
    h3_resolution_macro: int = 7   # Untuk Spatial Block Train-Test Split (~5 km²)
    h3_resolution_micro: int = 9   # Untuk Risk Grid Mapping (~0.1 km²)
    
    # Kata Kunci Pencarian Default untuk Wilayah Tangsel
    poi_queries: List[str] = field(default_factory=lambda: [
        "TPS Liar Tangerang Selatan",
        "Tempat Pembuangan Sampah Liar Ciputat",
        "TPS Liar Pamulang",
        "Sampah Liar Serpong Tangsel",
        "Lahan Kosong Sampah Pondok Aren"
    ])