"""
Utilities to load LangChain Documents
"""


import json
from pathlib import Path
from typing import Iterable

from langchain.schema import Document


def save_docs_to_jsonl(array: Iterable[Document], file_path: Path) -> None:
    """
    Save a list of documents in a JSONL file
    """
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + "\n")


def load_docs_from_jsonl(file_path: Path) -> Iterable[Document]:
    """
    Load a list of documents from a JSONL file
    """
    array = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
