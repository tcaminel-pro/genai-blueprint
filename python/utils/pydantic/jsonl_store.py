"""JSON Lines (JSONL) storage for Pydantic models.

Provides functions to load and store Pydantic models in JSONL format.
Each line contains a separate JSON object representing a model instance.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_objects_from_jsonl(file_path: Path, model_class: type[T]) -> list[T]:
    """Load objects from a JSONL file as Pydantic model instances.

    Args:
        file_path: Path to the JSONL file.
        model_class: Pydantic model class to instantiate objects.

    Returns:
        List of instantiated Pydantic model objects.
    """
    array = []
    logger.info(f"load {file_path} ")
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = model_class(**data)
            array.append(obj)
    return array


def store_objects_to_jsonl(objects: Iterable[BaseModel], file_path: Path) -> None:
    with open(file_path, "w") as jsonl_file:
        for doc in objects:
            jsonl_file.write(doc.model_dump_json(indent=None) + "\n")
