import hashlib
import logging
import re
import time
import zipfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from langdetect import detect, LangDetectException
import string
import random

# Configure default logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ---------------------------------------------------------------------------
# Path & Directory Configuration
# ---------------------------------------------------------------------------
# Assuming this script is located at 'internship/src/run.py'
# - Path(__file__).resolve() gets '/absolute/path/to/internship/src/run.py'
# - .parent gets '/absolute/path/to/internship/src'
# - .parent.parent gets '/absolute/path/to/internship' (the project root)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

class LockoutException(Exception):
    """Raised when the server blocks our IP address or rate-limits us."""
    pass

class BaseScraper:
    # Default schema - subclasses can override these class variables
    DEFAULT_COLUMNS: List[str] = ["page_link", "download_link", "downloaded"]
    ESSENTIAL_METADATA_COLUMNS: List[str] = ["page_link"]
    
    # Supported file extensions
    MIME_TO_EXTENSION: Dict[str, str] = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/msword": "doc",
        "image/jpeg": "jpg",
        "image/png": "png",
        "application/zip": "zip"
    }
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188"
    ]

    def __init__(
        self,
        source_name: str,
        base_folder: Union[str, Path] = "data",  # Relative to project root
        columns: Optional[List[str]] = None,
        essential_columns: Optional[List[str]] = None,
        metadata_save_batch_size: int = 100,
        download_save_batch_size: int = 50,
        use_tsv: bool = True,
        min_delay: float = 1.5,
        max_delay: float = 5.0
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay

        # Initialize failure counters
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

        self.logger = logging.getLogger(self.__class__.__name__)
        self.source_name = source_name
        
        # Safely resolve base_folder to an absolute path relative to PROJECT_ROOT
        # if a relative string or relative Path object is provided.
        resolved_base = Path(base_folder)
        if not resolved_base.is_absolute():
            self.base_folder = (PROJECT_ROOT / resolved_base).resolve()
        else:
            self.base_folder = resolved_base.resolve()
        
        # Configuration parameters
        self.metadata_save_batch_size = metadata_save_batch_size
        self.download_save_batch_size = download_save_batch_size
        self.use_tsv = use_tsv
        
        # Determine paths and separator
        self.separator = "\t" if use_tsv else ","
        ext = "tsv" if use_tsv else "csv"
        self.data_path = self.base_folder / "bronze" / self.source_name / f"{self.source_name}_metadata.{ext}"
        self.zip_path = self.base_folder / "bronze" / self.source_name / f"{self.source_name}_files.zip"
        
        # Dynamic Schema Configuration
        self.columns = columns or self.DEFAULT_COLUMNS
        self.essential_columns = essential_columns or self.ESSENTIAL_METADATA_COLUMNS
        
        # Ensure target data directories exist before loading or saving
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df = self._load_state()

        # Session Setup (for reuse, headers, and pooling)
        self.session = requests.Session()
        self.user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({"User-Agent": self.user_agent})

    def _load_state(self, force: bool = False) -> pd.DataFrame:
        """Loads existing state or initializes a new DataFrame using configured columns."""
        if self.data_path.exists() and not force:
            self.logger.info(f"Loading existing data state from {self.data_path}")
            try:
                df = pd.read_csv(self.data_path, sep=self.separator)
                # Ensure all configured columns are present
                for col in self.columns:
                    if col not in df.columns:
                        df[col] = None
                return df
            except Exception as e:
                self.logger.error(f"Error loading state: {e}. Starting fresh.")
        
        self.logger.info("Initializing new DataFrame.")
        return pd.DataFrame(columns=self.columns)

    def _save_state(self) -> None:
        """Saves current DataFrame state to file."""
        self.df.to_csv(self.data_path, sep=self.separator, index=False)
        self.logger.info(f"State saved to {self.data_path}")

    def _request(self, url: str, timeout: int = 30, retries: int = 3, **kwargs) -> Optional[requests.Response]:
        """Wrapper method with dynamic delays, 429 auto-backoff, and hard 403 exits."""
        # Track how many times we've paused for a 429 during this single request
        rate_limit_waits = 0
        max_rate_limit_waits = 2 

        for attempt in range(retries):
            delay = random.uniform(self.min_delay, self.max_delay)
            self.logger.debug(f"Sleeping for {delay:.2f} seconds before request...")
            time.sleep(delay)

            try:
                response = self.session.get(url, timeout=timeout, **kwargs)
                
                # --- FAILSAFE 1: Hard Firewall Block (Exit Immediately) ---
                if response.status_code == 403 or "captcha" in response.text.lower() or "cloudflare" in response.text.lower():
                    self.logger.critical(f"HARD BLOCK DETECTED on URL: {url} (Status: {response.status_code})")
                    raise LockoutException("The scraper was hard-blocked by the host server's firewall.")

                # --- FAILSAFE 2: Temporary Rate Limit (Pause & Wait) ---
                if response.status_code == 429:
                    rate_limit_waits += 1
                    if rate_limit_waits > max_rate_limit_waits:
                        raise LockoutException("Aborted: Hit the rate limit repeatedly on a single URL.")

                    # Look for the server's suggested wait time, default to 5 minutes (300s)
                    retry_after = response.headers.get("Retry-After")
                    wait_time = 300  # 5 minutes
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            pass
                    
                    self.logger.warning(
                        f"Rate limit (429) detected on {url}. "
                        f"Sleeping for {wait_time} seconds to let the block expire..."
                    )
                    time.sleep(wait_time)
                    continue  # Re-run this loop iteration to retry the request

                # Reset consecutive failure tracking on success
                self.consecutive_failures = 0
                
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                self.logger.warning(f"Request failed (Attempt {attempt + 1}/{retries}) for {url}: {e}")
                time.sleep(2.0)

        # Handle consecutive network timeouts/drops
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            raise LockoutException(f"Aborted: {self.consecutive_failures} consecutive network failures detected.")
            
        return None

    def _get_item_id(self, row: pd.Series) -> str:
        """Generates a safe unique identifier for file naming. Overridable by subclasses."""
        if "page_link" in row and pd.notna(row["page_link"]):
            return hashlib.md5(str(row["page_link"]).encode()).hexdigest()[:12]
        return hashlib.md5(str(row.name).encode()).hexdigest()[:12]

    def _sanitize_filename(self, name: str) -> str:
        """Removes illegal characters from a filename string."""
        return re.sub(r'[\\/*?:"<>|]', "", name)

    def _scrape_all_item_urls(self) -> None:
        raise NotImplementedError("Subclasses must implement _scrape_all_item_urls")

    def _scrape_item_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _scrape_item_metadata")

    def _download_file(self, download_url: str, filename_in_zip: str) -> bool:
        """Downloads a file and appends it to the zip archive."""
        response = self._request(download_url, stream=True)
        if not response:
            return False

        content_type = response.headers.get("Content-Type", "").lower()
        extension = None
        for mime, ext in self.MIME_TO_EXTENSION.items():
            if mime in content_type:
                extension = ext
                break

        if not extension:
            self.logger.warning(f"Unsupported Content-Type '{content_type}' for URL: {download_url}")
            return False

        full_filename = f"{filename_in_zip}.{extension}"
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(self.zip_path, "a", zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr(full_filename, response.content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write {full_filename} to ZIP: {e}")
            return False

    def run(self, gather_urls: bool = True, gather_metadata: bool = True, download_files: bool = True):
        self.logger.info(f"--- Starting scraper for: {self.source_name} ---")

        try:
            # Step 1: Gather URLs
            if gather_urls:
                self.logger.info("Step 1: Finding item URLs...")
                initial_count = len(self.df)
                self._scrape_all_item_urls()
                new_urls_count = len(self.df) - initial_count

                if new_urls_count > 0:
                    self.logger.info(f"Added {new_urls_count} new items to the dataframe.")
                    self._save_state()
                else:
                    self.logger.info("No new items found.")
            else:
                self.logger.info("Skipping Step 1: URL gathering.")

            # Step 2: Scrape Metadata
            if gather_metadata:
                self.logger.info("Step 2: Scraping missing metadata...")
                is_missing_metadata = self.df[self.essential_columns].isna().any(axis=1)
                rows_to_scrape = self.df[is_missing_metadata & self.df["page_link"].notna()]

                if rows_to_scrape.empty:
                    self.logger.info("No items require metadata scraping.")
                else:
                    self.logger.info(f"Found {len(rows_to_scrape)} items to scrape.")
                    batch = []

                    def _save_metadata_batch(current_batch: list):
                        if not current_batch:
                            return
                        self.logger.info(f"Saving batch of {len(current_batch)} metadata entries...")
                        update_df = pd.DataFrame(current_batch).set_index("original_index")

                        new_cols = update_df.columns.difference(self.df.columns)
                        if not new_cols.empty:
                            self.logger.info(f"Adding new cols to df {list(new_cols)}")
                            for col in new_cols:
                                self.df[col] = None

                        self.df = self.df.astype(object)
                        update_df = update_df.astype(object)

                        self.df.update(update_df)
                        self._save_state()

                    for idx, row in tqdm(rows_to_scrape.iterrows(), total=len(rows_to_scrape), desc="Scraping"):
                        metadata = self._scrape_item_metadata(row["page_link"])
                        if metadata:
                            metadata["original_index"] = idx
                            batch.append(metadata)

                        if len(batch) >= self.metadata_save_batch_size:
                            _save_metadata_batch(batch)
                            batch.clear()

                    if batch:
                        _save_metadata_batch(batch)
            else:
                self.logger.info("Skipping Step 2: Metadata scraping.")

            # Step 3: Download Files
            if download_files and "download_link" in self.df.columns:
                self.logger.info("Step 3: Checking for missing files...")
                existing_zip_files = set()
                if self.zip_path.exists():
                    with zipfile.ZipFile(self.zip_path, "r") as zipf:
                        existing_zip_files = {Path(f).stem for f in zipf.namelist()}

                # Mask items that have a download link but haven't been successfully downloaded yet
                mask = (self.df["download_link"].notna()) & (self.df["downloaded"] != True)
                to_download = self.df[mask].index

                if to_download.empty:
                    self.logger.info("No files to download.")
                else:
                    self.logger.info(f"Found {len(to_download)} items to download.")
                    success_count = 0
                    downloaded_this_batch = 0

                    for idx in tqdm(to_download, total=len(to_download), desc="Downloading"):
                        row = self.df.loc[idx]
                        
                        item_id = self._get_item_id(row)
                        safe_filename = self._sanitize_filename(item_id)

                        if safe_filename in existing_zip_files:
                            self.df.at[idx, "downloaded"] = True
                            success_count += 1
                            continue

                        if self._download_file(row["download_link"], safe_filename):
                            self.df.at[idx, "downloaded"] = True
                            existing_zip_files.add(safe_filename)
                            success_count += 1
                        else:
                            self.df.at[idx, "downloaded"] = False

                        downloaded_this_batch += 1

                        if downloaded_this_batch >= self.download_save_batch_size:
                            self.logger.info(f"Saving progress for {downloaded_this_batch} downloads...")
                            self._save_state()
                            downloaded_this_batch = 0

                    if downloaded_this_batch > 0:
                        self._save_state()

                    self.logger.info(f"Finished downloading. Added {success_count} new files.")
            else:
                self.logger.info("Skipping Step 3: File downloading.")

            self.logger.info(f"--- Scraper for {self.source_name} finished. ---")

        except LockoutException as e:
            self.logger.critical(f"FATAL EXCEPTION DETECTED: {e}")
            self.logger.info("Force-saving collected data to disk before exit...")
            self._save_state()
            sys.exit(1)


class HBOScraper(BaseScraper):
    DEFAULT_COLUMNS = ["download_link", 'page_link', "authors", 'abstract','downloaded','source']
    ESSENTIAL_METADATA_COLUMNS = ["abstract"]
    HEADING_WORDS = [
        "achtergrond", "inleiding", "doelstelling", "methode", "methoden", 
        "resultaat", "resultaten", "conclusie", "conclusies", "discussie", 
        "aanbeveling", "aanbevelingen", "samenvatting", "abstract", 
        "trefwoorden", "kernwoorden"
    ]

    def __init__(self, **kwargs):
        headings_list = []
        for h in self.HEADING_WORDS:
            headings_list.extend([h.lower(),h.capitalize(),h.upper()])
        self._headings_pattern = '|'.join(set(headings_list))
        super().__init__(
            source_name='HBO',
            columns=self.DEFAULT_COLUMNS,
            essential_columns=self.ESSENTIAL_METADATA_COLUMNS,
            use_tsv=False,
            **kwargs
        )

    def _get_item_id(self, row: pd.Series) -> str:
        author_slug = str(row["authors"]).lower().replace(" ", "_")
        return f"{author_slug}_{row.name}"
    
    def _scrape_all_item_urls(self) -> None:
        """Orchestrates multi-year, multi-organization traversal."""
        base_url = 'https://hbo-kennisbank.nl'
        
        # Iterate through every year from 2000 to 2022 separately
        for year in range(2000, 2023):
            self.logger.info(f"=========================================")
            self.logger.info(f"PROCESSING YEAR: {year}")
            self.logger.info(f"=========================================")
            
            # Formulate the initial page URL for the year (without organization filter)
            init_url = f"https://hbo-kennisbank.nl/searchresult?lng-0-u=dut&sort-order=date&date-from={year}&date-until={year}&t-0-k=hbo%3Aproduct&p=1"
            
            response = self._request(init_url)
            if not response:
                self.logger.warning(f"Failed to fetch initial page for year {year}. Skipping.")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the strong tag that holds the translation key for the search count
            strong_tag = soup.find("strong", attrs={"data-key": "num-results"})
            total_hits = 0
            
            if strong_tag:
                text_content = strong_tag.get_text(strip=True)  # e.g., "1.234 Resultaten"
                parts = text_content.split()
                if parts:
                    try:
                        raw_number = re.sub(r'[^\d]', '', parts[0])
                        total_hits = int(raw_number) if raw_number else 0
                    except ValueError:
                        self.logger.warning(f"Could not parse hit count from text: '{text_content}'")        
            
            self.logger.info(f"Year {year} total search hits: {total_hits}")
            
            if total_hits == 0:
                self.logger.info(f"No entries found for {year}. Moving to next year.")
                continue
                
            if total_hits <= 500:
                self.logger.info(f"Year {year} has <= 500 hits ({total_hits}). Scraping directly.")
                page_template = f"https://hbo-kennisbank.nl/searchresult?lng-0-u=dut&sort-order=date&date-from={year}&date-until={year}&t-0-k=hbo%3Aproduct&p={{page}}"
                self._scrape_pages_for_query(page_template, base_url)
            else:
                self.logger.info(f"Year {year} has > 500 hits ({total_hits}). Splitting by organization.")
                inputs = soup.select('div.search__aside__item__selection input[name="o"]')
                
                if not inputs:
                    inputs = soup.find_all("input", attrs={"name": "o"})
                    
                orgs = [inp.get("value") for inp in inputs if inp.get("value")]
                
                if not orgs:
                    self.logger.warning(
                        f"No organization inputs found for high-volume year {year}. "
                        f"Crawl is restricted to best-effort (first 50 pages)."
                    )
                    page_template = f"https://hbo-kennisbank.nl/searchresult?lng-0-u=dut&sort-order=date&date-from={year}&date-until={year}&t-0-k=hbo%3Aproduct&p={{page}}"
                    self._scrape_pages_for_query(page_template, base_url)
                    continue
                
                self.logger.info(f"Discovered {len(orgs)} organizations to partition year {year}: {orgs}")
                
                for org in orgs:
                    self.logger.info(f"-> Starting scrape for organization: '{org}' (Year: {year})")
                    org_template = f"https://hbo-kennisbank.nl/searchresult?lng-0-u=dut&sort-order=date&date-from={year}&date-until={year}&t-0-k=hbo%3Aproduct&o={org}&p={{page}}"
                    self._scrape_pages_for_query(org_template, base_url)

    def _scrape_pages_for_query(self, page_template: str, base_url: str) -> None:
        """Helper method to run a page-by-page crawl for a formatted template string."""
        page = 1
        patience = 0
        existing_urls = set(self.df['page_link'].dropna())
        previous_page_urls = set()

        while patience <= 3:
            url = page_template.format(page=page)
            self.logger.info(f'Fetching page {page}')
            response = self._request(url)
            if not response:
                patience += 1
                page += 1
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            tags = soup.find_all('a', class_='result-title')
            
            if not tags:
                self.logger.info(f"No results found on page {page}. Incrementing patience.")
                patience += 1
                page += 1
                continue
            
            current_page_urls = set()
            new_rows = []

            for tag in tags:
                href = tag.get("href")
                if not href:
                    continue
                full_url = f"{base_url}{href}" if href.startswith("/") else href
                current_page_urls.add(full_url)

            if previous_page_urls and previous_page_urls == current_page_urls:
                self.logger.warning(
                    f"Pagination ceiling reached on page {page} (identical page items). "
                    f"Breaking current query loop."
                )
                break
                
            previous_page_urls = current_page_urls
            patience = 0

            for full_url in current_page_urls:
                if full_url not in existing_urls:
                    new_rows.append({
                        "page_link": full_url,
                        "source": self.source_name,
                        "downloaded": False
                    })
                    existing_urls.add(full_url)

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.logger.info(f"Added {len(new_rows)} new URLs from page {page}.")
                self._save_state()
            
            page += 1

    def _scrape_item_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        # Define local cache paths first (Relative to the base folder we resolved)
        html_dir = self.base_folder / "raw_data" / self.source_name / "raw_html"
        html_dir.mkdir(parents=True, exist_ok=True)
        item_id = hashlib.md5(url.encode()).hexdigest()[:12]
        html_path = html_dir / f"{item_id}.html"

        html_content = None

        # Check if the raw HTML was already downloaded locally
        if html_path.exists():
            self.logger.info(f"Using locally cached HTML for {url}")
            try:
                html_content = html_path.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.error(f"Failed to read local HTML file {html_path}: {e}")

        # If no local copy exists, make the network request
        if not html_content:
            response = self._request(url)
            if not response:
                self.logger.warning(f"Skipping metadata scraping for {url} due to request failure.")
                return None
            html_content = response.text
            
            try:
                html_path.write_text(html_content, encoding="utf-8")
                self.logger.debug(f"Saved raw HTML to {html_path}")
            except Exception as e:
                self.logger.error(f"Failed to save raw HTML to disk: {e}")

        soup = BeautifulSoup(html_content, "html.parser")
        metadata: Dict[str, Any] = {}

        # Clean and save abstract (<p> child of <div class='detail__body'>)
        detail_body = soup.find("div", class_="detail__body")
        abstract = None
        if detail_body:
            p_tag = detail_body.find("p", recursive=False) or detail_body.find("p")
            if p_tag:
                abstract = p_tag.get_text(strip=True)
        metadata["abstract"] = abstract

        # Extract keywords
        keyword_tags = soup.select("a.detail__body__meta__list__item.detail__body__meta__list__item--label")
        keywords_list = [tag.get_text(strip=True) for tag in keyword_tags]
        metadata["keywords"] = "; ".join(keywords_list) if keywords_list else None

        # Extract metadata table inside <div class='detail__aside'>
        detail_aside = soup.find("div", class_="detail__aside")
        if detail_aside:
            for tr in detail_aside.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) == 2:
                    key = tds[0].get_text(strip=True)
                    val = tds[1].get_text(strip=True)
                    if key:
                        metadata[key.lower()] = val

        # Extract Title, Subtitle, and Authors inside <div class='detail__header__column ...'>
        header_column = soup.select_one('div[class*="detail__header__column"]')
        title = None
        subtitle = None
        authors_list = []

        if header_column:
            h1_tag = header_column.find("h1")
            if h1_tag:
                title = h1_tag.get_text(strip=True)
            
            subtitle_tag = header_column.find("span", class_="subtitle-text")
            if subtitle_tag:
                subtitle = subtitle_tag.get_text(strip=True)

            author_tags = header_column.find_all("a", class_="author-link")
            for a_tag in author_tags:
                author_text = a_tag.get_text(strip=True)
                cleaned_author = re.sub(r"\s*\(\s*Student\s*\)\s*", "", author_text, flags=re.IGNORECASE).strip()
                if cleaned_author:
                    authors_list.append(cleaned_author)

        metadata["title"] = title
        metadata["subtitle"] = subtitle
        metadata["authors"] = "; ".join(authors_list) if authors_list else None

        # Extract Download Link (<a class='detail__header__button'>)
        download_tag = soup.find("a", class_="detail__header__button")
        download_link = None
        if download_tag and download_tag.get("href"):
            href = download_tag["href"]
            base_url = "https://hbo-kennisbank.nl"
            download_link = f"{base_url}{href}" if href.startswith("/") else href
            
        metadata["download_link"] = download_link
        metadata['page_link'] = url
        if "datum" in metadata and metadata["datum"]:
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", metadata["datum"])
            if year_match:
                metadata["year"] = int(year_match.group(1))
            else:
                metadata["year"] = None

        if metadata['abstract']:
            metadata['abstract'] = self._clean_abstract(metadata['abstract'])
        return metadata
    
    def _strip_layout_headers(self, sent: str) -> tuple[str, Optional[str]]:
        orig = sent
        sent_cleaned = re.sub(r'[*_]{1,2}', '', orig).strip()
        
        sent_cleaned = re.sub(rf'^(?:{self._headings_pattern})([A-Z])', r'\1', sent_cleaned)
        sent_cleaned = re.sub(rf'^(?:{self._headings_pattern})[\s]*[:.-]+[\s]*', '', sent_cleaned)
        sent_cleaned = re.sub(rf'^(?:{self._headings_pattern})\s+([A-Z])', r'\1', sent_cleaned)
        
        if re.match(rf'^(?:{self._headings_pattern})$', sent_cleaned):
            sent_cleaned = ""

        if sent_cleaned != orig:
            if not sent_cleaned:
                removed = orig
            else:
                idx = orig.find(sent_cleaned)
                if idx != -1:
                    removed = orig[:idx]
                else:
                    removed = f"'{orig}' -> '{sent_cleaned}'"
            return sent_cleaned, removed
            
        return orig, None

    def _clean_abstract(
            self,
            abstract: str,
            min_char_length: int = 100,
            tokenizer_lang: str = 'dutch',
            detect_lang_tag: str = 'nl'
        ) -> str:
        
        dutch_abstract = ""
        if isinstance(abstract, str) and len(abstract) >= min_char_length and abstract.strip():
            abstract = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', abstract)
            
            raw_sentences = nltk.sent_tokenize(abstract, language=tokenizer_lang)
            cleaned_sentences = []
            
            for sent in raw_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                cleaned_sent, removed = self._strip_layout_headers(sent)
                
                if removed:
                    self.logger.debug(f"Stripped layout header: {repr(removed)}")
                
                if not cleaned_sent:
                    continue
                
                sent = cleaned_sent

                should_merge = False
                if cleaned_sentences:
                    if not re.match(r'^[A-Z]', sent):
                        should_merge = True
                    elif len(sent) >= 2 and sent[1] in string.punctuation:
                        should_merge = True
                
                if should_merge:
                    cleaned_sentences[-1] = cleaned_sentences[-1] + ' ' + sent
                else:
                    cleaned_sentences.append(sent)
            
            dutch_sentences = []
            for sent in cleaned_sentences:
                try:
                    if detect(sent) == detect_lang_tag:
                        dutch_sentences.append(sent)
                except LangDetectException:
                    continue
            if dutch_sentences:
                dutch_abstract = ' '.join(dutch_sentences)
        return dutch_abstract


class ScriptiebankScraper(BaseScraper):
    DEFAULT_COLUMNS = [
        "page_link",
        "download_link",
        "downloaded",
        "source",
        "title",
        "first_name",
        "last_name",
        "college",
        "year",
        "promoter",
        "themes",
        "keywords",
        "text_homepage"
    ]
    ESSENTIAL_METADATA_COLUMNS = ["title", "last_name"]

    def __init__(self, **kwargs):
        super().__init__(
            source_name="SB",
            columns=self.DEFAULT_COLUMNS,
            essential_columns=self.ESSENTIAL_METADATA_COLUMNS,
            use_tsv=False,
            **kwargs
        )
        self.base_url = 'https://scriptiebank.be'
        self.url_template = 'https://scriptiebank.be/?page={page_num}'
        self.thesis_url_pattern = re.compile(r"https://scriptiebank\.be/scriptie/\d{4}/[a-zA-Z0-9_:-]+")
        self.download_pattern = re.compile(r"/file/\d+/download\?token=[a-zA-Z0-9_-]+")

    def _get_item_id(self, row: pd.Series) -> str:
        author_slug = f"{str(row['first_name']).lower()}_{str(row['last_name']).lower()}".replace(" ", "_")
        return f"{author_slug}_{row.name}"

    def _scrape_all_item_urls(self) -> None:
        page = 0
        patience = 0
        existing_urls = set(self.df['page_link'].dropna())
        previous_page_urls = set()

        while patience <= 3:
            url = self.url_template.format(page_num=page)
            self.logger.info(f"Requesting page list: {url}")

            response = self._request(url)
            if not response:
                self.logger.warning(f"Failed to get page {page}. Skipping.")
                patience += 1
                page += 1
                continue

            found_urls_on_page = self.thesis_url_pattern.findall(response.text)
            current_page_urls = set(found_urls_on_page)

            if not current_page_urls:
                patience += 1
                self.logger.info(f"No URLs found on page {page}. Patience: {patience}/4")
                page += 1
                continue

            if previous_page_urls and previous_page_urls == current_page_urls:
                self.logger.warning(
                    f"Pagination ceiling reached on page {page} (identical page items). "
                    f"Breaking current query loop."
                )
                break

            previous_page_urls = current_page_urls
            patience = 0
            new_rows = []

            for full_url in current_page_urls:
                if full_url not in existing_urls:
                    new_rows.append({
                        "page_link": full_url,
                        "source": self.source_name,
                        "downloaded": False
                    })
                    existing_urls.add(full_url)

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.logger.info(f"Added {len(new_rows)} new URLs from page {page}.")
                self._save_state()

            page += 1
        
        self.logger.info(f"URL Crawl finished. Current unique entries in database: {len(self.df)}")

    def _scrape_item_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        # Define local cache paths first (Relative to the base folder we resolved)
        html_dir = self.base_folder / "raw_data" / self.source_name / "raw_html"
        html_dir.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        html_path = html_dir / f"{url_hash}.html"

        html_content = None

        # Check if the raw HTML was already downloaded locally
        if html_path.exists():
            self.logger.info(f"Using locally cached HTML for {url}")
            try:
                html_content = html_path.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.error(f"Failed to read local HTML file {html_path}: {e}")

        # If no local copy exists, make the network request
        if not html_content:
            response = self._request(url)
            if not response:
                self.logger.warning(f"Skipping metadata scraping for {url} due to request failure.")
                return None
            html_content = response.text
            
            try:
                html_path.write_text(html_content, encoding="utf-8")
                self.logger.debug(f"Saved raw HTML to {html_path}")
            except Exception as e:
                self.logger.error(f"Failed to save raw HTML to disk: {e}")

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            def _get_text_safe(element):
                return element.text.strip() if element else None

            def label_decompose(cont):
                if not cont: return None
                label = cont.find('div', class_='field-label-above')
                if label: label.decompose()
                return cont.get_text(strip=True)

            download_match = self.download_pattern.search(html_content)

            # Extract Promoters
            promoter_container = soup.select_one('div.thesis__promotors')
            promoters_list = []
            if promoter_container:
                label_div = promoter_container.find('div', class_='field-label-above')
                if label_div:
                    label_div.decompose()
                raw_names_string = promoter_container.get_text(strip=True)            
                if raw_names_string:
                    promoters_list = [name.strip() for name in raw_names_string.split(',')]

            metadata = {
                "title": _get_text_safe(soup.find("h1")),
                "first_name": _get_text_safe(soup.find("div", class_="thesis__first-name")),
                "last_name": _get_text_safe(soup.find("div", class_="thesis__last-name")),
                "college": label_decompose(soup.find("div", class_="thesis__college")),
                "year": int(label_decompose(soup.find("div", class_="thesis__year"))) if label_decompose(soup.find("div", class_="thesis__year")) else None,
                "promoter": "; ".join(promoters_list) if promoters_list else None,
                "themes": "; ".join([theme.get_text() for theme in soup.select('div.thesis__themes--item a')]) or None,
                "keywords": "; ".join([keyword.get_text() for keyword in soup.select('div.thesis__keywords--item a')]) or None,
                "text_homepage": " ".join([text for tag in soup.select('div.thesis__text p, div.thesis_text h3') if (text:=tag.get_text(strip=True))]) or None,
                "page_link": url,
                "download_link": self.base_url + download_match.group(0) if download_match else None
            }
            return metadata
        except Exception as e:
            self.logger.error(f"ERROR parsing metadata for {url}: {e}")
            return None