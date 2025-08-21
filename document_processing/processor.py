import os
import requests
from pathlib import Path
from typing import Optional, List, Union
import shutil

# --- Global Configuration ---
ORIGINAL_PAPERS_DIR = Path("original_paper_files")

class DocumentProcessingUtils:
    """
    A utility class for handling document downloads and file management.
    """

    def __init__(self):
        """Initializes the utility class and ensures directories exist."""
        ORIGINAL_PAPERS_DIR.mkdir(exist_ok=True)

    def download_file_from_url(self, url: str) -> Optional[Path]:
        """
        Downloads a file from a URL to the original_paper_files directory.
        Returns the local file path if successful.
        """
        try:
            file_name = url.split('/')[-1] or "downloaded_file.pdf"
            file_path = ORIGINAL_PAPERS_DIR / file_name
            
            if file_path.exists():
                print(f"File {file_name} already exists locally. Skipping download.")
                return file_path

            print(f"Downloading document from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Downloaded successfully to {file_path}")
            return file_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download document from URL: {e}")
            return None
        except Exception as e:
            print(f"An error occurred during file download: {e}")
            return None

    def process_documents(self, inputs: List[dict[str, Union[str, Path]]]):
        """
        Processes a list of URLs or uploaded files.
        """
        results = []
        for input_value in inputs:
            source_type = input_value.get("type")
            value = input_value.get("value")
            print(f"Processing input: {value}, source type: {source_type}")
            if not value:
                continue

            if source_type.lower() == "url":
                result = self.download_file_from_url(value)
                if result:
                    results.append(f"Successfully processed URL: {value}")
                else:
                    results.append(f"Failed to process URL: {value}")

            elif source_type.lower() == "pdf":
                file_path = Path(value)
                local_path = ORIGINAL_PAPERS_DIR / file_path.name
                print(f"Processing uploaded file: {local_path}")
                if not file_path.exists():
                    results.append(f"Error: Uploaded file not found at {file_path}")
                    continue

                if not local_path.exists():
                    shutil.copy(file_path, local_path)
                    results.append(f"Successfully processed and saved PDF: {file_path.name}")
                else:
                    results.append(f"PDF file {file_path.name} already exists. Skipping.")

        return "\n".join(results)