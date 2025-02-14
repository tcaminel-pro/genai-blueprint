"""API Test."""

from fastapi.testclient import TestClient

from src.fastapi_app import app
from src.utils.config_mngr import global_config

# Define your FastAPI routes and functions here

global_config().select_config("pytest")

client = TestClient(app)


def test_echo() -> None:
    response = client.get("/echo/hello")
    assert response.status_code == 200
    assert response.json() == {"msg": "hello"}
