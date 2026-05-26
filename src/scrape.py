import pandas as pd
import requests
import re
import time
import zipfile
import hashlib  # CHANGED: Added for creating safe, unique filenames for HTML/Files
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from mu import sanitize_filename, timed_request

class BaseScraper:
    def __init__(self, source_name: str, base_folder: str = "data"):
        self.source_name = source_name
        self.base_folder = Path(base_folder) # CHANGED: Store base_folder to allow subclasses to use it
        self.csv_path = self.base_folder / "raw_data" / self.source_name / "SB_metadata_raw.tsv"
        self.zip_path = self.base_folder / "raw_data" / self.source_name / "SB_files_raw.zip"
        self.csv_path.parent.mkdir(parents=True,exist_ok=True)
        self.df = self._load_state()
        self.metadata_save_batch_size = 100

    def _load_state(self, force=False):
        if self.csv_path.exists() and not force:
            print(f"Found existing data at {self.csv_path}. Loading state.")
            return pd.read_csv(self.csv_path)
        else:
            if not force:
                print("No existing data found. Starting fresh.")
            if force:
                print("Forcing creation of new dataframe.")
            return pd.DataFrame(columns=['title','first_name','last_name','college','year','promoter','themes','keywords','text_homepage','page_link','download_link', 'downloaded','source'])

    def _scrape_all_item_urls(self) -> None:
        raise NotImplementedError("Subclasses must implement _scrape_all_item_urls")

    def _scrape_item_metadata(self, url: str) -> dict | None:
        raise NotImplementedError("Subclasses must implement _scrape_item_metadata")

    def _download_file(self, download_url: str, filename_in_zip: str):
        mime_to_extension = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/msword": "doc"
        }
        
        response = timed_request(download_url, timeout=30)
        if not response:
            print(f"  -> Download failed (bad response) for {download_url}")
            return False

        content_type = response.headers.get("Content-Type", "").lower()
        extension = None
        for mime, ext in mime_to_extension.items():
            if mime in content_type:
                extension = ext
                break

        if not extension:
            print(f"  -> WARNING: Unsupported file type '{content_type}' for URL: {download_url}")
            return False

        full_filename = f"{filename_in_zip}.{extension}"
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(full_filename, response.content)
        return True

    def run(self, gather_urls: bool = True, gather_metadata: bool = True, download_files: bool = True):
        print(f"--- Starting scraper for: {self.source_name} ---")

        if gather_urls:
            print("Step 1: Finding all item URLs...")
            initial_count = len(self.df)
            self._scrape_all_item_urls()
            new_urls_count = len(self.df) - initial_count

            if new_urls_count > 0:
                print(f"  -> Added {new_urls_count} new items to the dataframe.")
                self.df.to_csv(self.csv_path, sep="\t", index=False)
                print(f"  -> State saved to {self.csv_path}")
                gather_metadata=True
            else:
                print("  -> No new items found.")
        else:
            print("Skipping Step 1: URL gathering.")

        if gather_metadata:
            print("\nStep 2: Scraping metadata for items missing it...")
            # CHANGED: Removed 'id' from essential_cols
            essential_cols = ['title','last_name','college','year','promoter']
            is_missing_metadata = self.df[essential_cols].isna().any(axis=1)
            rows_to_scrape = self.df[is_missing_metadata & self.df['page_link'].notna()]

            if rows_to_scrape.empty:
                print("  -> No items require metadata scraping.")
            else:
                print(f"  -> Found {len(rows_to_scrape)} items to scrape for metadata.")
                new_metadata_list = []

                def _save_metadata_batch(batch: list):
                    if not batch:
                        return
                    print(f"\nSaving batch of {len(batch)} metadata entries...")
                    update_df = pd.DataFrame(batch).set_index('original_index')
                    self.df.update(update_df)
                    if 'year' in self.df.columns:
                        # Ensure year is handled safely even if empty
                        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce').astype('Int64')
                    self.df.to_csv(self.csv_path, index=False, sep="\t")
                    print(f"  -> Progress saved to {self.csv_path}")

                for index, row in tqdm(rows_to_scrape.iterrows(), total=len(rows_to_scrape), desc="Scraping Metadata"):
                    metadata = self._scrape_item_metadata(row['page_link'])
                    if metadata:
                        metadata['original_index'] = index
                        new_metadata_list.append(metadata)
                    
                    if len(new_metadata_list) >= self.metadata_save_batch_size:
                        _save_metadata_batch(new_metadata_list)
                        new_metadata_list.clear()

                if new_metadata_list:
                    _save_metadata_batch(new_metadata_list)
        else:
            print("Skipping Step 2: Metadata scraping.")

        if download_files:
            print("\nStep 3: Checking for and downloading missing files...")
            existing_zip_files = set()
            if self.zip_path.exists():
                with zipfile.ZipFile(self.zip_path, 'r') as zipf:
                    existing_zip_files = set(Path(f).stem for f in zipf.namelist())

            mask = (self.df['download_link'].notna()) & (self.df['downloaded'] != True)
            to_download = self.df[mask].index
    
            if to_download.empty:
                print("  -> No files to download.")
            else:
                print(f"  -> Found {len(to_download)} items to download.")
                success_count = 0
                downloaded_this_batch = 0
                for index in tqdm(to_download, total=len(to_download), desc="Downloading Files"):
                    row = self.df.loc[index]
                    
                    # CHANGED: Since 'id' doesn't exist yet, we create a unique temporary 
                    # filename using a hash of the page_link to avoid collisions.
                    temp_id = hashlib.md5(row['page_link'].encode()).hexdigest()[:12]
                    safe_filename = sanitize_filename(temp_id)

                    if safe_filename in existing_zip_files:
                        self.df.loc[index,'downloaded'] = True
                        success_count += 1
                        continue

                    if self._download_file(row['download_link'], safe_filename):
                        success_count += 1
                        existing_zip_files.add(safe_filename)
                        self.df.loc[index, 'downloaded'] = True
                    else:
                        self.df.loc[index, 'downloaded'] = False

                    downloaded_this_batch += 1

                    if downloaded_this_batch >= 50:
                        print(f'\nSaving download progress for {downloaded_this_batch} items to {self.csv_path}...')
                        self.df.to_csv(self.csv_path, index=False, sep="\t")
                        downloaded_this_batch = 0
                
                if downloaded_this_batch > 0:
                    print(f'\nSaving final batch of {downloaded_this_batch} item to {self.csv_path}')
                    self.df.to_csv(self.csv_path, index=False, sep="\t")

                print(f"  -> Downloaded {success_count} new files.")
                self.df.to_csv(self.csv_path, index=False, sep="\t")
                print(f' -> state updated and saved to {self.csv_path}')
        else:
            print('Skipping Step 3: File downloading.')

        print(f"\n--- Scraper for {self.source_name} finished. ---")

class ScriptiebankScraper(BaseScraper):
    def __init__(self,source_name='SB',base_folder='data'):
        super().__init__(source_name="SB",base_folder='data')
        self.source_name = source_name
        self.base_folder = Path(base_folder)
        self.base_url = 'https://scriptiebank.be'
        self.url_template = 'https://scriptiebank.be/?page={page_num}'
        self.thesis_url_pattern = re.compile(r"https://scriptiebank\.be/scriptie/\d{4}/[a-zA-Z0-9_:-]+")
        self.download_pattern = re.compile(r"/file/\d+/download\?token=[a-zA-Z0-9_-]+")

    def _scrape_all_item_urls(self) -> None:
        all_found_urls = set()
        page = 0
        patience = 0
        while patience <= 3:
            url = self.url_template.format(page_num=page)
            print(f"Requesting page list: {url}")

            # CHANGED: Removed the hardcoded HTML saving from here. 
            # This step should only focus on finding the list of URLs.
            response = timed_request(url=url)
            if not response:
                print(f"  -> Failed to get page {page}. Skipping.")
                page += 1
                continue

            found_urls_on_page = self.thesis_url_pattern.findall(response.text)
            
            if not found_urls_on_page:
                patience += 1
                print(f"  -> No URLs found on page. Patience: {patience}/4")
            else:
                patience = 0
                pre_count = len(all_found_urls)
                all_found_urls.update(found_urls_on_page)
                print(f"  -> Discovered {len(all_found_urls) - pre_count} new unique URLs on this page.")
            page += 1
        
        print(f"\nDiscovered {len(all_found_urls)} total unique URLs from source.")
        existing_urls = set(self.df['page_link'].dropna())
        new_urls = sorted(list(all_found_urls - existing_urls))

        if new_urls:
            new_df = pd.DataFrame(new_urls, columns=['page_link'])
            new_df['source'] = self.source_name
            self.df = pd.concat([self.df, new_df], ignore_index=True)
    
    def _scrape_item_metadata(self, url: str) -> dict | None:
        response = timed_request(url)
        if not response:
            return None
        url_hash = hashlib.md5(url.encode()).hexdigest()
        html_save_path = self.base_folder / 'raw_data' / 'SB' / f"{url_hash}.html"
        html_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(html_save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except Exception as e:
            print(f"  -> WARNING: Could not save HTML for {url}: {e}")

        def _get_text_safe(element):
            return element.text.strip() if element else None

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            
            def label_decompose(cont):
                if not cont: return None
                label = cont.find('div', class_='field-label-above')
                if label: label.decompose()
                return cont.get_text(strip=True)

            download_match = self.download_pattern.search(response.text)

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
                'promoter': promoters_list,
                'themes': [theme.get_text() for theme in soup.select('div.thesis__themes--item a')],
                'keywords':[keyword.get_text() for keyword in soup.select('div.thesis__keywords--item a')],
                'text_homepage':[text for tag in soup.select('div.thesis__text p, div.thesis_text h3') if (text:=tag.get_text(strip=True))],
                "page_link": url,
                "download_link": self.base_url + download_match.group(0) if download_match else None
            }
            return metadata
        except Exception as e:
            print(f"  -> ERROR parsing metadata for {url}: {e}")
            return None