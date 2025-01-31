# Factrory for AutoGen models.
# WORK ONLY FOR OPENAI YET

from typing import cast

from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_openaiext_client import GroqChatCompletionClient

from python.ai_core.llm import LlmFactory


def get_autogen_model_from_llm_id(llm_id: str, **kwargs) -> ChatCompletionClient:
    factory = LlmFactory(
        llm_id=llm_id,
        llm_params=kwargs,
    )
    llm = factory.get()
    if factory.info.provider == "OpenAI":
        from langchain_openai import AzureChatOpenAI

        llm = cast(AzureChatOpenAI, llm)
        return OpenAIChatCompletionClient(
            model=llm.model_name,
            # api_key=llm.openai_api_key,
            # base_url=llm.openai_api_base,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            top_p=llm.top_p,
            frequency_penalty=llm.frequency_penalty,
            presence_penalty=llm.presence_penalty,
            timeout=llm.request_timeout,
            **kwargs,
        )

    elif factory.info.provider == "AzureOpenAI":
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from langchain_openai import AzureChatOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        llm_azure = cast(AzureChatOpenAI, llm)
        debug(llm_azure.name)
        # az_model_client = AzureOpenAIChatCompletionClient(
        #     # model=llm_azure.name,
        #     model="gpt-4o",
        #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        #     azure_deployment="gpt-4o",
        #     azure_ad_token_provider=token_provider,
        #     api_version="2024-02-15-preview",  # Not sure
        # )

        az_model_client = AzureOpenAIChatCompletionClient(
            azure_deployment="gpt-4o",
            model="gpt-4o",
            api_version="2024-06-01",
            azure_endpoint="https://{your-custom-endpoint}.openai.azure.com/",
            azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
            # api_key="sk-...", # For key-based authentication.
        )
        return az_model_client

    elif factory.info.provider == "Groq":
        # NOT FINISHED ,
        try:
            return GroqChatCompletionClient(model=llm.name)
        except KeyError as ex:
            raise ValueError(
                f"Cannot use llm-id '{llm_id}' to setup Autogen model - not in  GroqChatCompletionClient)"
            ) from ex
    else:
        raise ValueError(f"Cannot use llm-id '{llm_id}' to setup Autogen model")


if __name__ == "__main__":
    l = get_autogen_model_from_llm_id(llm_id="gpt_4o_openai")

    l = get_autogen_model_from_llm_id(llm_id="gpt_4o_azure")
    # l = get_autogen_model_from_llm_id(llm_id="llama33_70_groq")
    l = get_autogen_model_from_llm_id(llm_id="deepseek_chatv3_openrouter")
    # l = get_autogen_model_from_llm_id(llm_id="llama32_3_ollama")
    debug(l)
