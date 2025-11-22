''' download new '''
# scraper download only
from scrape import BaseScraper



scraper = BaseScraper(source_name='ugent')
scraper.source_name
scraper.run(gather_metadata=False,gather_urls=False,download_files=True)