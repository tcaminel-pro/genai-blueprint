"""API Test

"""

from fastapi.testclient import TestClient

from python.config import global_config
from python.fastapi_app import app

# Define your FastAPI routes and functions here

global_config().select_config("pytest")

client = TestClient(app)


def test_echo():
    response = client.get("/echo/hello")
    assert response.status_code == 200
    assert response.json() == {"msg": "hello"}
