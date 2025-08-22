from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging
class DocumentProcessor:
    """
    A class for processing documents, including downloading and organizing files.
    """

    def __init__(self, logger):
        self.logger = logger or logging.getLogger(__name__)
        # Default headers if not passed in
        self.headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    def batch_process_files(self, files: List) -> List[List]:
        """Processes a batch of files."""
        return [self.process_file(file) for file in files]

    def process_file(self, file) -> List:
        """Original processing logic with Docling"""
        try:
            if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
                print(f"Skipping unsupported file type: {file.name}")
                return []

            converter = DocumentConverter()
            markdown = converter.convert(file).document.export_to_markdown()
            # self.logger.info(f"Converted {file} to markdown.")
            # self.logger.info(f"Markdown content:\n{markdown}")
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers)
            return splitter.split_text(markdown)

        except Exception as e:
            self.logger.error(f"Error processing file {file}: {e}")
            return []
