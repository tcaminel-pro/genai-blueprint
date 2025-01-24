    
# See "ChatOpenAI": ("langchain_openai", "OPENAI_API_KEY"), 

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_openaiext_client import GeminiChatCompletionClient
from langchain_openai import ChatOpenAI

from python.ai_core.llm import LlmFactory, get_llm

def get_autogen_model_from_llm_id(llm_id: str, kwargs) -> OpenAIChatCompletionClient : 
    factory = LlmFactory(llm_id = llm_id)
    llm = get_llm(llm_id, **kwargs)  
    if isinstance(llm, ChatOpenAI) :
        # pass all attributes of 'llm' to OpenAIChatCompletionClient AI!
        return OpenAIChatCompletionClient()

