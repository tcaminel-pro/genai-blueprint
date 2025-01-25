"""
Entry point for the REST API

"""

from devtools import debug  # type: ignore  # noqa: F401
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger  # type: ignore

from python.ai_chains.A_1_joke import get_chain

load_dotenv(verbose=True)
app = FastAPI()


@app.get("/echo/{message}")
def read_root(message: str):
    logger.info(f"received /echo/{message}")
    return {"msg": message}


@app.post("/joke")
async def tell_a_joke(topic: str):
    """return a joke on a given topic"""
    result = get_chain({}).invoke(input={"topic": topic})
    return result


# TO BE COMPLETED !!
