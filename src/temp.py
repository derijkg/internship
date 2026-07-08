from scrape import HBOScraper
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

scraper = HBOScraper(base_folder= BASE_DIR / 'data')
scraper.run(gather_urls=False, gather_metadata=True, download_files=False)