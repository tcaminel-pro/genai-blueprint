"""
Smrrarize a conversation between a bot and a patient

Copyright (C) 2024 Eviden. All rights reserved
"""

import json
from pathlib import Path

from typing import List, Optional
from pydantic import BaseModel
from devtools import debug
from langchain_core.prompts import ChatPromptTemplate


# fmt: off
import sys
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

from python.ai.llm import get_llm


class WAConversationItem(BaseModel):
    """ Schema used by Watson Assistant"""
    a: Optional[str] = None
    u: Optional[str] = None
    n: Optional[bool] = None



PRE_PROMPT_SUMMARIZE = """
[INST]Voici un dialogue entre un médecin et un patient. En faire une synthèse en Francais du tableau du patient suivant le plan :
1. identité du patient
2. motif de la venue
3. signes cliniques
4. antécédents médicaux
5. traitements en cours
-----
{conversation}
[/INST]
"""


def summarize_chain(questions_anwsers : list[WAConversationItem]) ->str:
    """ Summarise a Watson Assistan conversation between the bot and a patient """

    conversation = ""
    for item in questions_anwsers:
        if item.n:
            pass
        if  item.a:
            conversation += f"\nMedecin: {item.a}"
        if  item.u:
            conversation += f"\nPatient: : {item.u}"

    prompt = ChatPromptTemplate.from_template(PRE_PROMPT_SUMMARIZE)
    chain = prompt | get_llm()
    summarized = chain.invoke({"conversation": conversation})
    return summarized
