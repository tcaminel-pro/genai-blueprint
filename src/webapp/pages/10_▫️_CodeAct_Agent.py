"""SmolAgents powered quite generic chat"""

from datetime import date
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
import yfinance as yf
from groq import BaseModel
from loguru import logger
from pydantic import ConfigDict
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    MCPClient,
    Tool,
    VisitWebpageTool,
    WebSearchTool,
    tool,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_folium import st_folium

from src.ai_core.llm import LlmFactory
from src.ai_core.mcp_client import dict_to_stdio_server_list, get_mcp_servers_dict
from src.ai_core.prompts import dedent_ws
from src.utils.streamlit.auto_scroll import scroll_to_here
from src.utils.streamlit.load_data import TABULAR_FILE_FORMATS_READERS, load_tabular_data
from src.utils.streamlit.recorder import StreamlitRecorder
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smolagents_streamlit import stream_to_streamlit

MODEL_ID = None  # Use the one by configuration
MODEL_ID = "gpt_41mini_openrouter"
# MODEL_ID = "qwen_qwq32_deepinfra"
# MODEL_ID = "gpt_o3mini_openrouter"
# MODEL_ID = "qwen_qwq32_openrouter"

DATA_PATH = Path.cwd() / "use_case_data/other"

if "agent_output" not in st.session_state:
    st.session_state.agent_output = []

if "result_display" not in st.session_state:
    st.session_state.result_display = st


class DataFrameTool(Tool):
    name: str
    description: str
    inputs = {
        "dataset": {
            "type": "string",
            "description": "data set required",
        }
    }
    output_type = "object"
    source_path: Path

    def __init__(self, name: str, description: str, source_path: Path) -> None:
        super().__init__()
        self.name = name
        self.description = f"This tool returns a Pandas DataFrame with content described as: '{description}'"
        self.source_path = source_path
        try:
            import pandas as pd  # noqa: F401
        except ImportError as e:
            raise ImportError("You must install package `pandas` to run this tool`.") from e

    def forward(self, dataset: str) -> pd.DataFrame:  # type: ignore
        df = get_cache_dataframe(self.source_path)
        return df


class Demo(BaseModel):
    name: str
    tools: list[Tool] = []
    mcp_servers: list[str] = []
    examples: list[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


SEARCH_TOOLS: list[Tool] = [WebSearchTool(), VisitWebpageTool()]


@tool
def get_stock_info(symbol: str, key: str) -> dict:
    """Return the correct stock info value given the appropriate symbol and key.
    If asked generically for 'stock price', use currentPrice.

    Args:
        symbol : Stock ticker symbol.
        key : must be one of the following: address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio.

    """
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]


@tool
def get_historical_price(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.

    Args:
        symbol (str): Stock ticker symbol.
        end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
        start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date.
    """
    try:
        data = yf.Ticker(symbol)
        hist = data.history(start=start_date, end=end_date)
        hist = hist.reset_index()
        hist[symbol] = hist["Close"]
        return hist[["Date", symbol]]
    except Exception as ex:
        logger.error(f"failed to call get_historical_price: {ex}")
        return pd.DataFrame()


##########################
#  SmolAgent parameters
##########################


AUTHORIZED_IMPORTS = [
    "pathlib",
    "pandas",
    "matplotlib.*",
    "numpy.*",
    "json",
    "streamlit",
    "base64",
    "tempfile",
    "sklearn.*",
    "folium.*",
    "wordcloud",
]

PRINT_INFORMATION = "my_final_answer"

IMAGE_INSTRUCTION = dedent_ws(
    f""" 
    -  When creating a plot or generating an image:
      -- save it as png in a tempory directory (use tempfile)
      -- call '{PRINT_INFORMATION}' with the pathlib.Path  
"""
)

FOLIUM_INSTRUCTION = dedent_ws(
    f""" 
    - If requested by the user, use Folium to display a map. For example: 
        -- to display map at a given location, call  folium.Map([latitude, longitude])
        -- Do your best to select the zoom factor so whole location enter globaly in the map 
        -- save it as png in a tempory directory (use tempfile)
        -- Call the function '{PRINT_INFORMATION}' with the map object 
        """
)

PRE_PROMPT = dedent_ws(
    f"""
    Answer following request. 

    Instructions:
    - You can use ONLY the following packages:  {", ".join(AUTHORIZED_IMPORTS)}.
    - DO NOT USE other packages (such as os, shutils, etc).
    - Don't generate "if __name__ == "__main__"
    - Don't use st.sidebar 
    - Call the function '{PRINT_INFORMATION}' with same content that 'final_answer', before calling it.

    - {FOLIUM_INSTRUCTION}
    - {IMAGE_INSTRUCTION}

    \nRequest :
    """
)

#  - Call the function '{PRINT_STEP}' to display intermediate informtation. It accepts markdown and str.
#  - Print also the outcome on stdio, or the title if it's a diagram.

##########################
#  Demos definition
#  todo : put it in a config file
##########################


SAMPLES_DEMOS = [
    Demo(
        name="Classic SmolAgents",
        tools=SEARCH_TOOLS,
        examples=[
            "How many seconds would it take for a leopard at full speed to run through Pont des Arts?",
            "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?",
            "Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men’s FIFA World Cup?",
        ],
    ),
    Demo(
        name="Titanic",
        tools=[
            DataFrameTool(
                name="titanic_data_reader",
                description="Data related to the Titanic passengers",
                source_path=DATA_PATH / "titanic.csv",
            )
        ]
        + SEARCH_TOOLS,
        examples=[
            "What is the proportion of female passengers that survived?",
            "Were there any notable individuals or families aboard ",
            "Plot in a bar chat the proportion of male and female survivors",
            "What was the survival rate of passengers on the Titanic?",
            "Did the passenger class have an impact on survival rates?",
            "What were the ticket fares and cabin locations for the passengers?"
            "What are the demographics (age, gender, etc.) of the passengers on the Titanic?",
            "What feature would you engineered to predict survival rate ? Build a predictive model, and report the F1 score on a test set",
        ],
    ),
    Demo(
        name="Stock Price",
        tools=[get_stock_info, get_historical_price] + SEARCH_TOOLS,
        examples=[
            "What is the current price of Meta stock?",
            "Show me the historical prices of Apple vs Microsoft stock over the past 6 months",
        ],
    ),
    Demo(
        name="Geo",
        tools=SEARCH_TOOLS,
        examples=[
            "Display the map of Toulouse",
            "Display a map of France with a population density (gathered from Internet) layer by region or department",
        ],
    ),
    Demo(
        name="MCP",
        tools=[],
        mcp_servers=["filesystem", "weather", "playwright"],
        examples=[
            "What is the currrent wind force in Toulouse ? ",
            "List current directory",
            "connect to atos.net site and get recent news",
        ],
    ),
]


##########################
#  UI
##########################

llm_config_widget(st.sidebar)


@st.cache_data(show_spinner=True)
def get_cache_dataframe(file_or_filename: Path | UploadedFile, **kwargs) -> pd.DataFrame:
    return load_tabular_data(file_or_filename=file_or_filename, **kwargs)


FILE_SElECT_CHOICE = ":open_file_folder: :orange[Select your file]"


strecorder = StreamlitRecorder()


def clear_display() -> None:
    st.session_state.agent_output = []
    strecorder.clear()
    # st.rerun()


c01, c02 = st.columns([6, 4], border=False, gap="medium", vertical_alignment="top")
c02.title(" CodeAct Agent :material/Mindfulness:")
with c01.container(border=True):
    selected_pill = st.pills(
        "🎬 **Demos:**",
        options=[demo.name for demo in SAMPLES_DEMOS] + [FILE_SElECT_CHOICE],
        default=SAMPLES_DEMOS[0].name,
        on_change=clear_display,
    )
raw_data_file = None
df: pd.DataFrame | None = None
sample_search = None


placeholder = st.empty()
select_block = placeholder.container()

if selected_pill == FILE_SElECT_CHOICE:
    raw_data_file = select_block.file_uploader(
        "Upload a Data file:",
        type=list(TABULAR_FILE_FORMATS_READERS.keys()),
        # on_change=clear_submit,
    )
    demo = Demo(name="custom", examples=[])
else:
    demo = next((d for d in SAMPLES_DEMOS if d.name == selected_pill), None)
    if demo is None:
        st.stop()

    col_display_left, col_display_right = select_block.columns([6, 3], vertical_alignment="bottom")
    with col_display_right:
        if tools_list := ", ".join(f"'{t.name}'" for t in demo.tools):
            st.markdown(f"**Tools**: *{tools_list}*")
        if mcp_list := ", ".join(f"'{mcp}'" for mcp in demo.mcp_servers):
            st.markdown(f"**MCP**: *{mcp_list}*")

    with col_display_left:
        # st.write("**Example Prompts:**")
        sample_search = col_display_left.selectbox(
            label="Sample",
            placeholder="Select an example (optional)",
            options=demo.examples,
            index=None,
            label_visibility="collapsed",
        )


if raw_data_file:
    with select_block.expander(label="Loaded Dataframe", expanded=True):
        args = {}
        df = get_cache_dataframe(raw_data_file, **args)


@tool
def my_final_answer(answer: Any) -> Any:
    """
    Provides a final answer to the given problem.

        Args:
        answer : The final answer to the problem.  Can be Markdown or an object of type pd.DataFrame, Path or folium.Map
    """
    # additional_args = getattr(my_final_answer, "_additional_args", {})
    # widget = additional_args.get("widget") # Does not woek here
    if len(st.session_state.agent_output) == 0 or st.session_state.agent_output[-1] != answer:
        st.session_state.agent_output.append(answer)
        display_final_msg(answer)
    return str(answer)


def update_display() -> None:
    if len(st.session_state.agent_output) > 0:
        st.write("answer:")
    for msg in st.session_state.agent_output:
        display_final_msg(msg)


def display_final_msg(msg: Any) -> None:
    try:
        with st.session_state.result_display:
            st.write(f"{type(msg)}")
            if isinstance(msg, str):
                st.markdown(msg)
            elif isinstance(msg, folium.Map):
                st_folium(msg)
            elif isinstance(msg, pd.DataFrame):
                st.dataframe(msg)
            elif isinstance(msg, Path):
                st.image(msg)
            else:
                st.write(msg)
    except Exception as ex:
        logger.exception(ex)
        raise ex


model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)


with select_block.form("my_form", border=False):
    cf1, cf2 = st.columns([15, 1], vertical_alignment="bottom")
    prompt = cf1.text_area(
        "Your task",
        height=68,
        placeholder="Enter or modify your query here...",
        value=sample_search or "",
        label_visibility="collapsed",
    )
    submitted = cf2.form_submit_button(label="", icon=":material/send:")

if submitted:
    HEIGHT = 800
    exec_block = placeholder.container()
    col_display_left, col_display_right = exec_block.columns(2)
    log_widget = col_display_left.container(height=HEIGHT)
    result_display = col_display_right.container(height=HEIGHT)
    # result_display = col_display_right

    st.session_state.result_display = result_display

    mcp_tools = []
    mcp_client = None

    try:
        if demo.mcp_servers:
            mcp_servers = dict_to_stdio_server_list(get_mcp_servers_dict(demo.mcp_servers))
            if mcp_servers:
                mcp_client = MCPClient(mcp_servers)  # type: ignore
                mcp_tools = mcp_client.get_tools()

        strecorder.replay(log_widget)
        with log_widget:
            if prompt:
                tools = demo.tools + mcp_tools + [my_final_answer]
                tools_list = [f"{t.name}: {t.description}" for t in tools]
                agent = CodeAgent(
                    tools=tools,
                    model=llm,
                    additional_authorized_imports=AUTHORIZED_IMPORTS,
                    max_steps=10,  # for debug
                )
                debug(agent.tools)
                with st.spinner(text="Thinking..."):
                    result_display.write(f"query: {prompt}")
                    with strecorder:
                        stream_to_streamlit(
                            agent,
                            PRE_PROMPT + prompt,
                            # additional_args={"widget": col_display_right},  # does not work in fact
                            display_details=False,
                        )
                    scroll_to_here()
            with col_display_right:
                update_display()
    finally:
        if mcp_client:
            mcp_client.disconnect()
