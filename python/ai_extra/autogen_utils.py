    
# See "ChatOpenAI": ("langchain_openai", "OPENAI_API_KEY"), 

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_openaiext_client import GeminiChatCompletionClient
from langchain_openai import ChatOpenAI

from python.ai_core.llm import LlmFactory, get_llm

def get_autogen_model_from_llm_id(llm_id: str, kwargs) -> OpenAIChatCompletionClient : 
    factory = LlmFactory(llm_id = llm_id)
    llm = get_llm(llm_id, **kwargs)  
    if isinstance(llm, ChatOpenAI) :
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
            **kwargs
        )

