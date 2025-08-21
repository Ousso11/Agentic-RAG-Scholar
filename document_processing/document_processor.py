from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

class DocumentProcessor:
    """
    A class for processing documents, including downloading and organizing files.
    """
    def batch_process_files(self, files: List) -> List[List]:
        """Processes a batch of files."""
        return [self.process_file(file) for file in files]

    def process_file(self, file) -> List:
        """Original processing logic with Docling"""
        if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
            print(f"Skipping unsupported file type: {file.name}")
            return []

        converter = DocumentConverter()
        markdown = converter.convert(file.name).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)