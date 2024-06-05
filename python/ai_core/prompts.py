""" 
Wrapper and utilities around prompts.


"""

from textwrap import dedent
from typing import Any, Optional, cast

from langchain.schema import SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from loguru import logger
from pydantic import BaseModel

from python.ai_core.llm import get_llm

DEFAULT_SYSTEM_PROMPT = ""


def def_prompt(system: str | None = None, user: str = "") -> BasePromptTemplate:
    """
    Small wrapper around 'ChatPromptTemplate.from_messages" with just a user  and optional system prompt.
    """
    messages: list = []
    if system:
        messages.append(("system", dedent(system)))
    messages.append(("user", dedent(user)))

    return ChatPromptTemplate.from_messages(messages)


class PromptFormatter(BaseModel):
    """
    Format a list off Messages to a prompt adapted to LLM specific format.
    Supported : LLama2, LLama3, Mistral, ...
    Inspired by : https://python.langchain.com/docs/integrations/chat/llama2_chat/
      (https://api.python.langchain.com/en/latest/chat_models/langchain_experimental.chat_models.llm_wrapper.Llama2Chat.html )
    but without the burden to create a new LLM

    NOT FULLY TESTED - WIP...
    """

    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

    text_beg: str = ""
    sys_beg: str
    sys_end: str
    ai_n_beg: str
    ai_n_end: str
    usr_n_beg: str
    usr_n_end: str
    usr_0_beg: Optional[str] = None
    usr_0_end: Optional[str] = None
    text_end: str = ""

    def to_chat_prompt_template(
        self,
        input_prompt: Any,
    ) -> BasePromptTemplate:
        if isinstance(input_prompt, str):
            input_prompt = def_prompt(user=input_prompt)
            # input_prompt = def_prompt(user=input_prompt)
        elif not isinstance(input_prompt, ChatPromptTemplate):
            logger.warning(
                f"ignore formatting of message of type : {type(input_prompt)}"
            )
            return input_prompt

        if self.usr_0_beg is None:
            self.usr_0_beg = self.usr_n_beg

        if self.usr_0_end is None:
            self.usr_0_end = self.usr_n_end

        template = []
        template.append(self.text_beg)
        for message in input_prompt.messages:
            if isinstance(message, SystemMessagePromptTemplate):
                template.append(
                    self.sys_beg
                    + dedent(cast(str, message.prompt.template) + self.sys_end)  # type: ignore
                )
            elif isinstance(message, HumanMessagePromptTemplate):
                template.append(
                    self.usr_n_beg
                    + dedent(cast(str, message.prompt.template) + self.usr_n_end)  # type: ignore
                )
            else:
                raise Exception(f"cannot transform {message} in single prompt")
        template.append(self.text_end)
        new_template = "".join(template)
        return PromptTemplate(
            template=new_template, input_variables=input_prompt.input_variables
        )

    def to_chat_msg(
        self,
        input_msg: Any,
    ):
        template = ChatPromptTemplate.from_messages(input_msg)
        return self.to_chat_prompt_template(template)


class Llama2Format(PromptFormatter):
    """Format LLama2
    NOT TESTED
    """

    sys_beg: str = "<s>[INST] <<SYS>>\n"
    sys_end: str = "\n<</SYS>>\n\n"
    ai_n_beg: str = " "
    ai_n_end: str = " </s>"
    usr_n_beg: str = "<s>[INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = ""
    usr_0_end: str = " [/INST]"


class Llama3Format(PromptFormatter):
    """Wrapper for Llama-3-chat model.
    see  https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#meta-llama-3-chat
    """

    text_beg: str = "<|begin_of_text|>"
    sys_beg: str = "<|start_header_id|>system<|end_header_id|>"
    sys_end: str = "<|eot_id|>\n"
    ai_n_beg: str = "<|start_header_id|>assistant<|end_header_id|>"
    ai_n_end: str = "<|eot_id|>"
    usr_n_beg: str = "<|start_header_id|>user<|end_header_id|>"
    usr_n_end: str = "<|eot_id|>"
    text_end: str = "<|start_header_id|>assistant<|end_header_id|>"


class MixtralFormat(PromptFormatter):
    """See https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#instruction-format"""  # noqa: E501

    # NOT TESTED - see usr_0 and usr_N stuff
    sys_beg: str = "<s>[INST] "
    sys_end: str = "\n"
    ai_n_beg: str = " "
    ai_n_end: str = " </s>"
    usr_n_beg: str = " [INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = ""
    usr_0_end: str = " [/INST]"


llama3_formatter = RunnableLambda(lambda x: Llama3Format().to_chat_prompt_template(x))
mistral_formatter = RunnableLambda(lambda x: MixtralFormat().to_chat_prompt_template(x))


def test():
    from langchain.prompts import PromptTemplate

    formatter = Llama3Format()

    prompt_original = PromptTemplate(
        template=""" <|begin_of_text|><| start_header_id|>system<|end_header_id|> this is a system message talking about {topic} <|eot_id|>
    <|start_header_id|>user<|end_header_id|> <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["topic"],
    )

    prompt = def_prompt(
        system="this is a system message talking about {topic}",
        user="This is a message",
    )

    new_prompt = formatter.to_chat_prompt_template(prompt)

    # debug(prompt_original, prompt, new_prompt)
    llama3_local = get_llm(llm_id="llama3_8_local")
    chain = new_prompt | llama3_local
    chain.invoke("french")


if __name__ == "__main__":
    test()
