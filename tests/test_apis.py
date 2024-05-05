"""
API Test

Copyright (C) 2023 Eviden. All rights reserved
"""

import json
import sys
from pathlib import Path

from devtools import debug
from fastapi import FastAPI
from fastapi.testclient import TestClient

# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on


from python.fastapi_app import app

# Define your FastAPI routes and functions here

client = TestClient(app)


def test_echo():
    response = client.get("/echo/hello")
    assert response.status_code == 200
    assert response.json() == {"msg": "hello"}
