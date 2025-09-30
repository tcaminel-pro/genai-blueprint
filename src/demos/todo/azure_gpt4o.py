from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


# model = "gpt4-turbo/2023-05-15"  # WORKS
model = "gpt-4o/2023-05-15"  # DOES NOT WORK
name, _, api_version = model.partition("/")
llm = AzureChatOpenAI(
    name=name,
    azure_deployment=name,
    model=name,  # Not sure it's needed
    api_version=api_version,
)

r = llm.invoke("tell me a joke")
print(r)
