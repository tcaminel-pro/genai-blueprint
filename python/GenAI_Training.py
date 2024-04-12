from functools import cached_property
import os, sys
from pathlib import Path
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain.agents import (
    load_tools,
    AgentExecutor,
    create_structured_chat_agent,
    create_openai_tools_agent,
)
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain import hub
from lunary import LunaryCallbackHandler


# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on

from python.st_utils.authentication import authenticate


EVIDEN_LOGO = str(Path.cwd() / "static" / "eviden-logo-white.png")

class AppConfig():
    chat_gpt: BaseLanguageModel 
    embeddings_model: Embeddings | None = None
    lunary_api_key: str | None = None
    use_functions: bool = True
    use_cache : bool = True  

    @cached_property
    def callback_handlers (self) -> list[BaseCallbackHandler]:
        if self.lunary_api_key:
            return [LunaryCallbackHandler(app_id=self.lunary_api_key)]
        else:
            return []


    class Config:
        arbitrary_types_allowed = True

@st.cache_resource(show_spinner=True)
def app_conf() -> AppConfig:
    return AppConfig()


api_base = (
    "https://poc-mw.openai.azure.com/openai"  # was "https://poc-mw.openai.azure.com/",
)


# fmt: off
st.set_page_config(
    page_title="LLM Powered Agents Demos",
    page_icon="🛠️", layout="wide" ,initial_sidebar_state="expanded")
# fmt:on

# debug(os.environ.get("BASIC_AUTHENTICATION", "0"))

# os.environ["BASIC_AUTHENTICATION"] = "0"  # for tests
if os.environ.get("BASIC_AUTHENTICATION", "0") != "0":
    st.session_state["authenticated"] = authenticate()
else:
    st.session_state["authenticated"] = True

if st.session_state.get("authenticated"):
    st.sidebar.success("Select a demo above.")

    title_col1, title_col2 = st.columns([2, 1])

    title_col2.image(EVIDEN_LOGO, width=250)
    title_col1.markdown(
        f"""
        ## Agents Demos <sub>({os.getenv("VERSION","dev")})</sub>.

        LLM powered Agents and Multi Agents are a powerful AI Technique.<br>

        **👈 Select a demo from the sidebar** to see some examples""",
        unsafe_allow_html=True,
    )


def config_sidebar():
    if not st.session_state.get("authenticated"):
        st.write("not authenticated")
        return
    with st.sidebar:
        with st.expander("LLM Configuration", expanded=False):
            end_point = st.radio(
                label="LLM Endpoint", options=["Azure", "OpenAI"], index=1
            )
            if end_point == "OpenAI":
                app_conf().chat_gpt = ChatOpenAI(
                    model="gpt-3.5-turbo-1106",
                    temperature=0.0,
                    model_kwargs={"seed": 42},
                )  # type: ignore

                app_conf().embeddings_model = OpenAIEmbeddings()

            elif end_point == "Azure":
                os.environ["AZURE_OPENAI_ENDPOINT"] = "https://poc-mw.openai.azure.com/"
                app_conf().embeddings_model = AzureOpenAIEmbeddings(
                    azure_deployment="ds-team-text-embedding-ada-002",  # type: ignore
                    openai_api_version="2023-05-15",  # type: ignore
                )
                app_conf().chat_gpt = AzureChatOpenAI(
                    temperature=0.0,
                    azure_deployment="ds-team-gpt-35-turbo",  # type: ignore
                    openai_api_version="2023-05-15",  # type: ignore
                )
            app_conf().use_functions = st.checkbox(
                label="Use functions",
                value=True,
                help="Use OpenAI style functions calls (available on recent models)",
            )

            app_conf().use_cache = st.checkbox(
                label="Use LLM cache",
                value=True,
                help="Use cached LLM responses",
            )
            cache = (
                SQLiteCache(database_path=".langchain.db")
                if app_conf().use_cache
                else None
            )
            set_llm_cache(cache)

            api_key = None
            if st.checkbox(
                label="Use Lunary.ai for monitoring",
                value=True,
                help="Lunary.ai is a LLM monitoring service. It's free until 1K event/day ",
            ):
                api_key = os.getenv("LUNARY_APP_ID")
                if not api_key:
                    api_key = st.text_input(
                        label="Lunary.ai API Key",
                        type="password",
                        help="[LLM Monitor](https:lunary.ai/) API Key not set. Please provide yours",
                    )
                    if api_key:
                        os.environ["LUNARY_APP_ID"] = api_key
                        os.environ["LLMONITOR_APP_ID"] = api_key
            app_conf().lunary_api_key = api_key
            if app_conf().lunary_api_key:
                text = "See [Lunary](https://app.lunary.ai/logs?type=trace) for LLM calls and tool use"
                st.markdown(text)

            if st.button(label="LLM Agent Test"):
                import numexpr

                result = llm_agent_test(
                    "What is the approximate result of 3 to the power of 4?"
                )
                st.write(result)


def llm_agent_test(query):
    assert app_conf().chat_gpt
    tools = load_tools(["llm-math"], llm=app_conf().chat_gpt)
    if app_conf().use_functions:
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(
            prompt=prompt,
            tools=tools,
            llm=app_conf().chat_gpt,
        )
    else:
        prompt = hub.pull("hwchase17/structured-chat-agent")
        agent = create_structured_chat_agent(
            prompt=prompt, tools=tools, llm=app_conf().chat_gpt
        )
    output_container = st.sidebar.empty()
    output_container.write("callback:")
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=app_conf().callback_handlers
        + [StreamlitCallbackHandler(output_container)],
        metadata={"agentName": f"math test  {datetime.now().strftime('%d/%m-%H:%M')}"},
    )
    result = agent_executor.invoke({"input": query})

    return result
