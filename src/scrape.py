import pandas as pd
import requests
import re
import time
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from mu import sanitize_filename_new, timed_request

class BaseScraper:
    """A base class for scraping theses from different sources.

    This class provides a framework for scraping, including state management
    (loading/saving a metadata CSV), file downloading, and a structured
    execution flow. Subclasses must implement the methods for finding
    item URLs and scraping metadata for a single item.

    :param source_name: The name of the data source (e.g., 'scriptiebank').
        This is used to create a dedicated folder for the source's data.
    :param base_folder: The root folder where all scraper data will be stored.
        Defaults to "data".
    """
    def __init__(self, source_name: str, base_folder: str = "data"):
        self.source_name = source_name
        base_folder = Path(base_folder)
        self.csv_path = base_folder / "metadata.csv"
        self.zip_path = base_folder / "archive.zip"
        self.df = self._load_state()
        self.metadata_save_batch_size = 100


    def _load_state(self, force=False):
        """Loads the existing metadata CSV into a pandas DataFrame.

        If the CSV file specified by `self.csv_path` exists, it is loaded.
        Otherwise, an empty DataFrame with a 'page_link' column is created.

        :return: A pandas DataFrame containing the loaded or initial state.
        """
        if self.csv_path.exists() and not force:
            print(f"Found existing data at {self.csv_path}. Loading state.")
            return pd.read_csv(self.csv_path)
        else:
            if not force:
                print("No existing data found. Starting fresh.")
            if force:
                print("Forcing creation of new dataframe.")
            return pd.DataFrame(columns=['id','title','first_name','last_name','college','year','promoter','themes','keywords','text_homepage','page_link','download_link', 'downloaded','source'])


    def _scrape_all_item_urls(self) -> None:                                    # specific
        """[Abstract Method] Scrapes the source for all individual item URLs.

        Subclasses must implement this method to find all relevant item pages
        (e.g., individual thesis pages) from the source. It should compare these
        found URLs with the ones already in `self.df` and add any new URLs.
        After this method runs, `self.df` should contain all unique item URLs.
        """
        raise NotImplementedError("Subclasses must implement _scrape_all_item_urls")


    def _scrape_item_metadata(self, url: str) -> dict | None:                   # specific
        """[Abstract Method] Scrapes metadata from a single item page.

        Subclasses must implement this method to extract metadata (like title,
        author, year, download link) from a given item page URL.

        :param url: The URL of the item page to scrape.
        :return: A dictionary containing the scraped metadata, or None if
            scraping fails.
        """
        raise NotImplementedError("Subclasses must implement _scrape_item_metadata")

    def _download_file(self, download_url: str, filename_in_zip: str):
        """Downloads a file and writes it to the source's zip archive.

        The method attempts to download the content from `download_url`. It infers
        the file extension from the response's Content-Type header and saves the
        file into the zip archive defined by `self.zip_path`.

        :param download_url: The direct URL to the file to be downloaded.
        :param filename_in_zip: The base name for the file inside the zip archive
            (without the extension).
        :return: True if the download and save were successful, False otherwise.
        """
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
        with zipfile.ZipFile(self.zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(full_filename, response.content)
        return True


    def run(self, gather_urls: bool = True, gather_metadata: bool = True, download_files: bool = True):
        """Main execution loop for the scraper.

        This method orchestrates the entire scraping process, which consists of
        three optional steps:
        1. Gather all item URLs.
        2. Scrape metadata for items missing it.
        3. Download files for items that have a download link.

        :param gather_urls: If True, executes the URL gathering step.
        :param gather_metadata: If True, executes the metadata scraping step.
        :param download_files: If True, executes the file downloading step.
        """
        print(f"--- Starting scraper for: {self.source_name} ---")

        # 1. Gather all item URLs and add new ones to the DataFrame
        if gather_urls:
            print("Step 1: Finding all item URLs...")
            initial_count = len(self.df)
            self._scrape_all_item_urls()
            new_urls_count = len(self.df) - initial_count

            if new_urls_count > 0:
                print(f"  -> Added {new_urls_count} new items to the dataframe.")
                self.df.to_csv(self.csv_path, index=False)
                print(f"  -> State saved to {self.csv_path}")
                print(f'Turning on gather_metadata since new urls have been found.')
                gather_metadata=True
            else:
                print("  -> No new items found.")
        else:
            print("Skipping Step 1: URL gathering.")

        # 2. Scrape metadata for items that are missing it
        if gather_metadata:
            print("\nStep 2: Scraping metadata for items missing it...")
            essential_cols = ['title','last_name','college','year','keywords','promoter']
            is_missing_metadata = self.df[essential_cols].isna().any(axis=1)
            rows_to_scrape = self.df[is_missing_metadata & self.df['page_link'].notna()]

            if rows_to_scrape.empty:
                print("  -> No items require metadata scraping.")
            else:
                print(f"  -> Found {len(rows_to_scrape)} items to scrape for metadata.")
                new_metadata_list = []

                def _save_metadata_batch(batch: list):
                    """Helper function to update the DataFrame and save to CSV."""
                    if not batch:
                        return
                    # The progress bar can be momentarily interrupted by printing, so we add newlines.
                    print(f"\nSaving batch of {len(batch)} metadata entries...")
                    update_df = pd.DataFrame(batch).set_index('original_index')
                    self.df.update(update_df)
                    if 'year' in self.df.columns:
                        self.df['year'] = self.df['year'].astype('Int64')
                    self.df.to_csv(self.csv_path, index=False)
                    print(f"  -> Progress saved to {self.csv_path}")

                for index, row in tqdm(rows_to_scrape.iterrows(), total=len(rows_to_scrape), desc="Scraping Metadata"):
                    metadata = self._scrape_item_metadata(row['page_link'])
                    if metadata:
                        metadata['original_index'] = index
                        new_metadata_list.append(metadata)
                    
                    # Check if the batch is full and needs to be saved
                    if len(new_metadata_list) >= self.metadata_save_batch_size:
                        _save_metadata_batch(new_metadata_list)
                        new_metadata_list.clear() # Reset the list for the next batch

                # Save any remaining items that didn't make up a full batch
                if new_metadata_list:
                    _save_metadata_batch(new_metadata_list)

        else:
            print("Skipping Step 2: Metadata scraping.")


        # 3. Download missing files
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
                    base_filename = f"{row.get('id', 'NA')}"
                    safe_filename = sanitize_filename_new(base_filename)

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
                        self.df.to_csv(self.csv_path, index=False)
                        downloaded_this_batch = 0
                if downloaded_this_batch > 0:
                    print(f'\nSaving final batch of {downloaded_this_batch} item to {self.csv_path}')
                    self.df.to_csv(self.csv_path, index=False)

                
                print(f"  -> Downloaded {success_count} new files.")
                self.df.to_csv(self.csv_path, index=False)
                print(f' -> state updated and saved to {self.csv_path}')
        else:
            print('Skipping Step 3: File downloading.')

        print(f"\n--- Scraper for {self.source_name} finished. ---")

class ScriptiebankScraper(BaseScraper):
    """A scraper for theses from scriptiebank.be.

    This class implements the specific logic required to paginate through
    the Scriptiebank website, find all individual thesis pages, and extract
    their metadata and download links.
    :param source_name: The name of the data source (e.g., 'scriptiebank').
        This is used to create a dedicated folder for the source's data.
    :param base_folder: The root folder where all scraper data will be stored.
        Defaults to "data".
    """
    def __init__(self):
        """Initializes the ScriptiebankScraper.

        Sets the source name to 'scriptiebank' and configures the URLs and
        regex patterns specific to the Scriptiebank website structure.
        """
        super().__init__(source_name="ugent")
        # inherits:
        #self.source_name = source_name
        #self.csv_path = base_folder / "metadata.csv"
        #self.zip_path = base_folder / "archive.zip"
        #self.df = self._load_state()
        #self.metadata_save_batch_size = 100

        self.base_url = 'https://scriptiebank.be'
        self.url_template = 'https://scriptiebank.be/?page={page_num}'
        self.thesis_url_pattern = re.compile(r"https://scriptiebank\.be/scriptie/\d{4}/[a-zA-Z0-9_:-]+")
        self.download_pattern = re.compile(r"/file/\d+/download\?token=[a-zA-Z0-9_-]+")

    def _scrape_all_item_urls(self) -> None:
        """Scrapes all thesis page URLs from scriptiebank.be.

        It iterates through the paginated list of theses, extracts all
        links matching the thesis URL pattern, and adds new, unique URLs
        to the main DataFrame. The process stops after several consecutive
        pages yield no new URLs.
        """
        all_found_urls = set()
        page = 0
        patience = 0
        while patience <= 3:
            url = self.url_template.format(page_num=page)
            print(f"Requesting page list: {url}")
            response = timed_request(url)
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
            last_id = -1
            if not self.df.empty and 'id' in self.df.columns and self.df['id'].notna().any():
                last_id = self.df['id'].max()
            new_df['id'] = range(int(last_id)+1, int(last_id)+1+len(new_df))
            self.df = pd.concat([self.df, new_df], ignore_index=True)
    
    def _scrape_item_metadata(self, url: str) -> dict | None:
        """Scrapes metadata from a single Scriptiebank thesis page.

        Parses the HTML of the given thesis page to extract the title,
        author's name, college, year, and a direct download link for the file.

        :param url: The URL of the Scriptiebank thesis page.
        :return: A dictionary of the scraped metadata, or None if parsing fails.
        """
        response = timed_request(url)
        if not response:
            return None
        
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
                "year": int(label_decompose(soup.find("div", class_="thesis__year"))),
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

class BiblioScraper(BaseScraper):
    """
    A scraper for theses from the Ghent University Academic Bibliography API.
    This scraper is designed to be more efficient by fetching all metadata
    from a single API endpoint rather than scraping individual HTML pages.
    """
    def __init__(self):
        super().__init__(source_name="biblio_ugent")
        self.api_url = "https://biblio.ugent.be/publication"
        self.query = 'type=dissertation AND accesslevel="info:eu-repo/semantics/openAccess"'

    def _scrape_all_item_urls(self) -> None:
        """
        Fetches all open access dissertation records from the Biblio API
        and populates the DataFrame with their full metadata at once.
        This overrides the typical "URL gathering" step with a more direct
        and efficient data retrieval method.
        """

        print(f"Requesting all records from Biblio API...")
        params = {'q': self.query, 'format': 'json', 'n': 10000} # n=10000 is the max limit
        response = timed_request(self.api_url, params=params)
        
        if not response:
            print("  -> Failed to get a response from the Biblio API.")
            return

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("  -> Failed to decode JSON from the Biblio API response.")
            return
        
        print(f"  -> API returned {len(data)} records.")
        print(data)
        processed_records = []
        for item in tqdm(data, desc="Processing API records"):
            # --- Extract Author ---
            author = item.get('author', [{}])[0]
            first_name = author.get('first_name')
            last_name = author.get('last_name')

            # --- Extract Promoter(s) ---
            promoters_list = [f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                              for p in item.get('promoter', [])]
            
            # --- Extract Download Link ---
            download_link = None
            for file_info in item.get('file', []):
                if file_info.get('kind') == 'fullText' and file_info.get('access') == 'open':
                    download_link = file_info.get('url')
                    break # Take the first available open access full text file

            # --- Assemble Metadata ---
            record = {
                'id': item.get('_id'),
                'title': item.get('title'),
                'first_name': first_name,
                'last_name': last_name,
                'college': None,  # Not consistently available in this API endpoint
                'year': item.get('year'),
                'promoters': ", ".join(promoters_list) if promoters_list else None,
                'page_link': f"https://biblio.ugent.be/publication/{item.get('_id')}",
                'download_link': download_link
            }
            processed_records.append(record)

        if not processed_records:
            print("  -> No new records to add.")
            return

        # Create a DataFrame from the new records
        new_df = pd.DataFrame(processed_records)

        # Merge with existing data, avoiding duplicates
        if not self.df.empty and 'id' in self.df.columns:
            # Filter out records that are already in the dataframe
            existing_ids = set(self.df['id'].dropna())
            new_df = new_df[~new_df['id'].isin(existing_ids)]

        if not new_df.empty:
            self.df = pd.concat([self.df, new_df], ignore_index=True)


    def _scrape_item_metadata(self, url: str) -> dict | None:
        """
        This method is not needed for the BiblioScraper because all metadata
        is retrieved in a single batch from the API in _scrape_all_item_urls.
        It is implemented to satisfy the abstract base class requirement.
        """
        return None
    

# deprecated
class GentScraper(BaseScraper):
    """A scraper for theses from the Ghent University Library (lib.ugent.be).

    This class implements the logic to scrape thesis data from the UGent
    library catalog. It is designed to handle the pagination of search results
    and extract links to metadata files.
    :param source_name: The name of the data source (e.g., 'scriptiebank').
        This is used to create a dedicated folder for the source's data.
    :param base_folder: The root folder where all scraper data will be stored.
        Defaults to "data".
    """
    def __init__(self):
        """Initializes the GentScraper.

        Sets the source name to 'ugent' and configures the URL template for
        searching the UGent library catalog.
        """
        super().__init__(source_name="ugent")
        self.url_template = 'https://lib.ugent.be/en/catalog?access=online&lang=dut-eng-und&max_year=2022&min_year=1980&page={page_num}&sort=old-to-new&type=bachelor-master-other-phd'

    def _scrape_all_item_urls(self) -> None:
        """Scrapes all item metadata JSON URLs from the UGent catalog.

        Iterates through paginated search results, parsing each page to find
        links to publicly accessible theses. It extracts a direct link to a
        JSON file containing the thesis metadata and adds new, unique URLs
        to the DataFrame. It handles restricted-access items by skipping them.
        """
        all_found_urls = set()
        
        def handle_href(href):
            pattern = re.compile(r'([a-zA-Z]{3}\d{2}:\d+)/')
            match = pattern.search(href)
            if match:
                return f'https://lib.ugent.be/catalog/{match.group(1)}.json'
            print('-> No id found.')
            return None
        
        page = 0
        patience = 0
        while patience <= 10:
            found_restricted = False

            url = self.url_template.format(page_num=page)
            print(f"Requesting page list: {url}")
            response = timed_request(url)
            if not response:
                print(f"  -> Failed to get page {page}. Skipping and increasing patience {patience}.")
                patience += 1
                page += 1
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            documents = soup.find('div', class_='documents-list')
            if documents:
                found_urls_on_page = set()
                for doc in documents.find_all('div', recursive=False):
                    if doc.find('small', class_='help-block text-center'):
                        found_restricted = True
                        continue
                    elif (ul := doc.find('ul', class_='dropdown-menu pull-right')):
                        found_restricted = True
                        for li in ul.find_all('li'):
                            if 'ugent only' in li.text.lower(): continue
                            if not (link_tag := li.find('a')): continue
                            if not (href := link_tag.get('href')): continue
                            if (meta_link := handle_href(href)):
                                found_urls_on_page.add(meta_link)
                        else: continue
                    elif (link_tag:= doc.find('a', class_='btn btn-primary btn-block')):
                        if not (href:= link_tag.get('href')): continue
                        if (meta_link:= handle_href(href)):
                            found_urls_on_page.add(meta_link)
                    else:
                        print(f"WARNING: Unforeseen document structure on page {page}. Skipping document.")

                if not found_urls_on_page and not found_restricted:
                        patience += 1
                        print(f"  -> No URLs found on page. Patience: {patience}/10")
                else:
                    patience = 0
                    pre_count = len(all_found_urls)
                    all_found_urls.update(found_urls_on_page)
                    if len(all_found_urls) > pre_count:
                        print(f"  -> Discovered {len(all_found_urls) - pre_count} new unique URLs on this page.")
            page += 1
        
        print(f"\nDiscovered {len(all_found_urls)} total unique URLs from source.")
        existing_urls = set(self.df['page_link'].dropna())
        new_urls = sorted(list(all_found_urls - existing_urls))

        if new_urls:
            new_df = pd.DataFrame(new_urls, columns=['page_link'])
            self.df = pd.concat([self.df, new_df], ignore_index=True)
    

    def _scrape_item_metadata(self, url: str) -> dict | None:
        """Scrapes metadata from a UGent JSON data source.

        This method fetches and parses the JSON file at the given URL to
        extract detailed thesis metadata. It uses .get() for safe key access
        to prevent errors from missing data.

        :param url: The URL of the metadata JSON file.
        :return: A dictionary containing the scraped metadata, or None if
            parsing fails or essential data is missing.
        """
        response = timed_request(url)
        if not response:
            return None

        try:
            data = response.json()
            # All useful information is nested inside these keys
            doc = data.get('response', {}).get('document', {})

            if not doc:
                print(f"  -> ERROR: 'document' key not found in JSON for {url}")
                return None
            
            # --- Extract Author Name(s) ---
            # The 'author' field is often "First Last", which is easy to split.
            author_list = doc.get('author', [])
            first_name, last_name = None, None
            if author_list:
                # We'll just use the first author listed
                full_name = author_list[0].strip()
                name_parts = full_name.split()
                if len(name_parts) > 1:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:])
                else:
                    # If there's only one word, assume it's the last name
                    last_name = full_name
            
            # --- Extract College/Faculty ---
            # 'display_corp_author' provides the full faculty name
            corp_author_list = doc.get('display_corp_author', [])
            college = corp_author_list[0] if corp_author_list else None
            
            # --- Extract Download Link ---
            # The 'files' list contains potential links. We need one that is not
            # private or restricted. Note: the link is often an SFX resolver,
            # not a direct PDF link.
            download_link = None
            for file_item in doc.get('files', []):
                if file_item.get('is_private') == 'false' and file_item.get('is_restricted') == 'false':
                    download_link = file_item.get('url')
                    if download_link:
                        break # Found a usable link, stop searching
            
            # --- Assemble Metadata Dictionary ---
            metadata = {
                "title": doc.get('title'),
                "first_name": first_name,
                "last_name": last_name,
                "college": college,
                "year": doc.get('year'),
                "page_link": url,  # The JSON URL itself is the unique identifier
                "download_link": download_link
            }
            
            # Ensure we have at least a title to consider it a valid entry
            if not metadata['download_link']:
                return None

            return metadata

        except (requests.exceptions.JSONDecodeError, KeyError) as e:
            print(f"  -> ERROR parsing metadata for {url}: Invalid JSON or key error - {e}")
            return None