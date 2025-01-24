# See "ChatOpenAI": ("langchain_openai", "OPENAI_API_KEY"),

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_openaiext_client import GroqChatCompletionClient
from langchain_openai import ChatOpenAI

from python.ai_core.llm import LlmFactory


def get_autogen_model_from_llm_id(llm_id: str, **kwargs) -> OpenAIChatCompletionClient:
    factory = LlmFactory(
        llm_id=llm_id,
        llm_params=kwargs,
    )
    llm = factory.get()
    debug(isinstance(llm, ChatOpenAI))
    if isinstance(llm, ChatOpenAI):
        return OpenAIChatCompletionClient(
            model=llm.model_name,
            api_key=llm.openai_api_key,
            base_url=llm.openai_api_base,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            top_p=llm.top_p,
            frequency_penalty=llm.frequency_penalty,
            presence_penalty=llm.presence_penalty,
            timeout=llm.request_timeout,
            **kwargs,
        )
    elif factory.info.cls == "ChatGroq":
        # NOT FINISHED ,
        try :
            return GroqChatCompletionClient(model=llm.name)
        except KeyError: 
            raise  ValueError(f"Cannot use llm-id '{llm_id}' to setup Autogen model - not in  GroqChatCompletionClient)")
    else:
        raise ValueError(f"Cannot use llm-id '{llm_id}' to setup Autogen model")


if __name__ == "__main__":
    l = get_autogen_model_from_llm_id(llm_id="gpt_4o_openai")
   # l = get_autogen_model_from_llm_id(llm_id="llama33_70_groq")
    l = get_autogen_model_from_llm_id(llm_id="deepseek_v3_openrouter")
    l = get_autogen_model_from_llm_id(llm_id="llama32_3_ollama")
    debug(l)
