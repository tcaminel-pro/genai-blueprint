import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

TABULAR_FILE_FORMATS = {
    "csv": pd.read_csv,
    # "xls": pd.read_excel,  # Could be read but need additional import
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
}


def load_tabular_data(file_or_filename: Path | UploadedFile) -> pd.DataFrame | None:
    if isinstance(file_or_filename, Path):
        assert file_or_filename.exists
        with open(file_or_filename, "rb") as file:
            loaded_file = BytesIO(file.read())
        loaded_file.name = file_or_filename.name
    elif isinstance(file_or_filename, UploadedFile):
        loaded_file = file_or_filename
    else:
        raise ValueError("incorrect file")
    ext = os.path.splitext(loaded_file.name)[1][1:].lower()
    if ext in TABULAR_FILE_FORMATS:
        return TABULAR_FILE_FORMATS[ext](loaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None
