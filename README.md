# GeoAI Open Burning Tangsel Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-WebGIS-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **End-to-End GeoAI Causal Framework** designed to model bare land vulnerability, quantify illegal open waste burning risks, estimate stochastic carbon emissions using IPCC Tier-2 methodologies, and optimize waste infrastructure location-allocation (P-Median) in South Tangerang Regency (Tangsel), Indonesia.

---

## Key Features

- **Spatial Causal Inference (Double Machine Learning / DML):** Uses Sentinel-2 indices (Bare Soil Index - BSI, NDVI, NDBI) alongside spatial confounders to isolate pure causal treatment effects on open burning susceptibility with standard errors ($p$-value reporting).
- **Uber H3 Hexagonal Spatial Partitioning:** Full-coverage resolution-8 spatial gridding for continuous regional risk intensity mapping.
- **IPCC Tier-2 Monte Carlo Emission Uncertainty Simulation:** 10,000-iteration stochastic Engine estimating annual $\text{CO}_2\text{e}$ equivalent emissions with $95\%$ Confidence Interval bounds.
- **P-Median Facility Location Optimization:** Allocates optimized TPS3R (Resource Recovery Center) candidate sites to minimize spatial transport impedance for high-risk zones.
- **Interactive WebGIS Dashboard:** Built using Streamlit and Folium Leaflet with seamless Google Satellite and OpenStreetMap switching.

---

## Repository Structure

```text
geoai-openburning-tangsel/
├── app/
│   └── streamlit_app.py         # Main Interactive Streamlit WebGIS Dashboard
├── data/
│   ├── figures/                 # PDF/PNG Publication Charts & WebGIS HTML Map
│   ├── processed/               # Cleaned & Partitioned Geospatial Data
│   └── raw/                     # Raw POI & Sentinel-2 Spectral Indices
├── src/
│   ├── ingestion/               # Geocoding & Nominatim Spatial Extractors
│   ├── models/                  # Causal DML & Monte Carlo Stochastic Engines
│   ├── optimization/            # P-Median Facility Allocator
│   └── visualization/           # Spatial Risk Maps & Folium Interactive Generators
├── pipeline_runner.py           # End-to-End Execution Orchestrator
├── requirements.txt             # Python Dependencies
├── .gitignore                   # Git Protection Rules
└── README.md                    # Project Documentation