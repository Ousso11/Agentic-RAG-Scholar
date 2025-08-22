import os
import json
import hashlib
import logging
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path

class DocumentProcessor:
    """
    A class for processing documents with caching support.
    Converts files to markdown and reuses cached conversions if unchanged.
    """

    def __init__(self, logger=None, cache_dir: str = ".cache"):
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Cache index file
        self.index_file = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}

        # Default headers for splitting
        self.headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    def _get_file_hash(self, file) -> str:
        """Compute SHA256 hash from a file path or file-like object."""
        hasher = hashlib.sha256()

        if isinstance(file, (str, Path)):  # if it's a path
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        else:  # file-like (has read/seek)
            file.seek(0)
            for chunk in iter(lambda: file.read(8192), b""):
                hasher.update(chunk)
            file.seek(0)

        return hasher.hexdigest()

    def _save_cache(self, file_hash: str, markdown: str, file_name: str):
        """Save markdown to cache and update index."""
        cache_md_path = os.path.join(self.cache_dir, f"{file_hash}.md")

        # Save markdown
        with open(cache_md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        # Update index
        self.cache_index[file_name] = {
            "hash": file_hash,
            "md_file": cache_md_path,
        }
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.cache_index, f, indent=2)

    def _load_cache(self, file_name: str, file_hash: str) -> str | None:
        """Check cache index and return cached markdown if valid."""
        entry = self.cache_index.get(file_name)
        if entry and entry["hash"] == file_hash and os.path.exists(entry["md_file"]):
            self.logger.info(f"Loaded from cache: {file_name}")
            with open(entry["md_file"], "r", encoding="utf-8") as f:
                return f.read()
        return None

    def batch_process_files(self, files: List) -> List[List]:
        """Processes a batch of files with caching."""
        return [self.process_file(file) for file in files]

    
    def process_file(self, file) -> List:
        """Process a single file with caching support."""
        try:
            file_path = Path(file) if isinstance(file, (str, Path)) else Path(file.name)

            if not file_path.suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
                self.logger.warning(f"Skipping unsupported file type: {file_path.name}")
                return []

            file_hash = self._get_file_hash(file)

            # Try cache
            markdown = self._load_cache(file_path.name, file_hash)
            if not markdown:
                self.logger.info(f"Converting {file_path.name} to markdown...")
                converter = DocumentConverter()
                # Use proper path if file is a Path
                if isinstance(file, (str, Path)):
                    markdown = converter.convert(file_path).document.export_to_markdown()
                else:
                    markdown = converter.convert(file).document.export_to_markdown()

                self._save_cache(file_hash, markdown, file_path.name)

            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers)
            return splitter.split_text(markdown)

        except Exception as e:
            self.logger.error(f"Error processing file {getattr(file, 'name', file)}: {e}")
            return []
