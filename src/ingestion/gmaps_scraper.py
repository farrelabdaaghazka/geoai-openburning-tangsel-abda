import asyncio
import logging
import re
from typing import List, Dict, Any
import pandas as pd
from playwright.async_api import async_playwright
from config.spatial_config import SpatialConfig

logger = logging.getLogger(__name__)

async def scrape_gmaps_poi(queries: List[str], config: SpatialConfig) -> pd.DataFrame:
    """Mengekstrak lokasi potensial TPS Liar dari Google Maps menggunakan Playwright."""
    results: List[Dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            channel="chrome",  
            args=["--no-sandbox", "--disable-setuid-sandbox", "--lang=id-ID"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for query in queries:
            url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
            try:
                # Gunakan domcontentloaded agar tidak hang menunggu background network Google Maps
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(3000)  # Delay stabilisasi DOM

                elements = await page.query_selector_all('a[href*="/maps/place/"]')

                for elem in elements:
                    href = await elem.get_attribute("href") or ""
                    name = await elem.get_attribute("aria-label") or "Unlabeled_POI"

                    match = re.search(r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)', href) or re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', href)

                    if match:
                        lat, lon = float(match.group(1)), float(match.group(2))

                        if config.min_lon <= lon <= config.max_lon and config.min_lat <= lat <= config.max_lat:
                            results.append({
                                "source": "Google_Maps_POI",
                                "location_name": name,
                                "latitude": lat,
                                "longitude": lon,
                                "confidence_score": 0.90
                            })
            except Exception as err:
                logger.warning(f"Gagal memproses query '{query}': {err}")
                continue

        await browser.close()

    df = pd.DataFrame(results)
    return df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True) if not df.empty else df