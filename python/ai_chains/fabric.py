#  Chain to call patterns (i.e. prompt + markdown) from the "fabric" project
#  Prompts are here : https://github.com/danielmiessler/fabric/tree/main/patterns
#
#  Code inspired from : https://github.com/danielmiessler/fabric/tree/main?tab=readme-ov-file#directly-calling-patterns

import re

import requests
from devtools import debug  # noqa: F401
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    chain,
)

from python.ai_core.chain_registry import Example, RunnableItem, register_runnable
from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt


# Pull the URL content's from the GitHub repo
def fetch_content_from_url(url):
    """Fetches content from the given URL."""

    ALLOWLIST_PATTERN = re.compile(r"^[a-zA-Z0-9\s.,;:!?\-]+$")

    try:
        response = requests.get(url)
        response.raise_for_status()
        sanitized_content = "".join(
            char for char in response.text if ALLOWLIST_PATTERN.match(char)
        )
        return sanitized_content
    except requests.RequestException:
        return ""


@chain
def fabric_prompt(param: dict):
    """Fetch the pattern from the Fabric GitHub web site and return a prompt.

    Argument is a dict with 2 keys: pattern name, and input date
    """

    URL = "https://raw.githubusercontent.com/danielmiessler/fabric/main/patterns/"
    URL = "https://raw.githubusercontent.com/danielmiessler/fabric/refs/heads/main/patterns/"
    system_url = f"{URL}/{param['pattern']}/system.md"
    user_url = f"{URL}/{param['pattern']}/user.md"
    # Fetch the prompt content
    system_content = fetch_content_from_url(system_url)
    user_file_content = fetch_content_from_url(user_url)

    return def_prompt(
        system=system_content, user=user_file_content + f"\n{param['input_data']}"
    )


def get_fabric_chain(config: dict):
    chain = (
        RunnablePassthrough()
        | fabric_prompt
        | get_llm(llm_id=config["llm"])
        | StrOutputParser()
    )
    return chain


# DOES NOT WORK (yet : need handling several args)
register_runnable(
    RunnableItem(
        tag="Fabric",
        name="Fabric pattern",
        runnable=get_fabric_chain,
        examples=[
            Example(
                query=[""],
            )
        ],
    )
)

# For quick test
if __name__ == "__main__":
    set_verbose(True)
    set_debug(True)
    load_dotenv(verbose=True)
    r = get_fabric_chain(config={"llm": None}).invoke(
        {"pattern": "ai", "input_data": "tell me more about stoicism"}
    )
    debug(r)
