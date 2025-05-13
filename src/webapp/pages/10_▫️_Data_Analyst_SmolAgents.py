"""AI Chat interface for analyzing data" """

from datetime import date
from pathlib import Path
from typing import Any, Sequence

import folium
import pandas as pd
import smolagents.default_tools
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
from src.ai_core.prompts import dedent_ws
from src.ai_extra.mcp_client import dict_to_stdio_server_list, get_mcp_servers_dict
from src.utils.streamlit.auto_scroll import scroll_to_here
from src.utils.streamlit.load_data import TABULAR_FILE_FORMATS_READERS, load_tabular_data
from src.utils.streamlit.recorder import StreamlitRecorder
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smolagents_streamlit import stream_to_streamlit

MODEL_ID = None  # Use the one by configuration
# MODEL_ID = "qwen_qwq32_deepinfra"
# MODEL_ID = "gpt_o3mini_openrouter"
# MODEL_ID = "qwen_qwq32_openrouter"

DATA_PATH = Path.cwd() / "use_case_data/other"

if "agent_output" not in st.session_state:
    st.session_state.agent_output = []


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
    tools: Sequence[Tool] = []
    mcp_servers: Sequence[str] = []
    examples: list[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


SEARCH_TOOLS = [WebSearchTool(), VisitWebpageTool()]


# cSpell: disable
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
    "numpy",
    "json",
    "streamlit",
    "base64",
    "tempfile",
    "sklearn.*",
    "folium.*",
    "requests.*",
    "wordcloud"
]

FINAL_FUNCTION = "print_result"

FOLIUM_INSTRUCTION = dedent_ws(
    """ 
    - Use Folium to display a map. For example: 
        -- to display map at a given location, call  folium.Map([latitude, longitude])
        -- Do your best to select the zoom factor so whole location enter globaly in the map
        -- Call the function '{FINAL_FUNCTION}' with the map object
        -- 
"""
)

IMAGE_INSTRUCTION = dedent_ws(
    f""" 
    -  When creating a plot or generating an image:
      -- save it as png in a tempory directory (use tempfile)
      -- call {FINAL_FUNCTION} with the pathlib.Path  
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
#   - Call the function '{FINAL_FUNCTION}' to display the final result. It accepts markdown (first choice), str, number, or a pathlib.Path to a generated image, or whenever possible  Python objects of Pandas Dataframe, or Follium Map.
#   - Print also the outcome on stdio, or the title if it's a diagram.

    - {FOLIUM_INSTRUCTION}
    - {IMAGE_INSTRUCTION}

    \nRequest :
    """
)


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
]


##########################
#  UI
##########################

st.title("Analytics AI Agent")
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


selected_pill = st.pills(
    "🎬 **Demos:**",
    options=[demo.name for demo in SAMPLES_DEMOS] + [FILE_SElECT_CHOICE],
    default=SAMPLES_DEMOS[0].name,
    on_change=clear_display,
)


raw_data_file = None
df: pd.DataFrame | None = None
tools = []
sample_search = None

if selected_pill == FILE_SElECT_CHOICE:
    raw_data_file = st.file_uploader(
        "Upload a Data file:",
        type=list(TABULAR_FILE_FORMATS_READERS.keys()),
        # on_change=clear_submit,
    )
else:
    demo = next((d for d in SAMPLES_DEMOS if d.name == selected_pill), None)
    if demo is None:
        st.stop()
    tools = demo.tools

    col1, answer_widget = st.columns([3, 1])
    with answer_widget:
        txt = ", ".join(f"'{t.name}'" for t in tools)
        st.write("Tools: " + txt)

    with col1:
        # st.write("**Example Prompts:**")
        sample_search = col1.selectbox("Sample queries:", demo.examples, index=None)


if raw_data_file:
    with st.expander(label="Loaded Dataframe", expanded=True):
        args = {}
        df = get_cache_dataframe(raw_data_file, **args)


class MyFinalAnswerTool(smolagents.default_tools.FinalAnswerTool):
    """
    adds session state tracking for displaying answers in the Streamlit UI.
    """

    def forward(self, answer: Any) -> Any:
        st.session_state.agent_output.append(answer)
        super().forward(answer)


# Replace the default FinalAnswerTool with our custom version
# smolagents.default_tools.FinalAnswerTool = MyFinalAnswerTool


class DisplayAnswerTool(Tool):
    name = "print_result"
    description = "Display important step in the reasonning (1 sentence) or the final answer to the given query."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        try:
            if len(st.session_state.agent_output) == 0 or st.session_state.agent_output[-1] != answer:
                st.session_state.agent_output.append(answer)
            return f"answer displayed: {answer}"
        except Exception as ex:
            logger.exception(ex)
            raise ex


def update_display() -> None:
    try:
        if len(st.session_state.agent_output) > 0:
            st.write("answer:")
        for msg in st.session_state.agent_output:
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

with st.form("my_form"):
    prompt = st.text_area(
        "Your task", height=68, placeholder=" query here...", value=sample_search or "", label_visibility="collapsed"
    )
    submitted = st.form_submit_button("GO !", disabled=prompt is None)

if submitted:
    col1, answer_widget = st.columns(2)
    log_widget = col1.container(height=400)

    mcp_tools = []
    mcp_client = None

    try:
        mcp_servers = dict_to_stdio_server_list(get_mcp_servers_dict(["filesystem", "weather"]))
        if mcp_servers:
            mcp_client = MCPClient(mcp_servers)  # type: ignore
            mcp_tools = mcp_client.get_tools()

        strecorder.replay(log_widget)
        with log_widget:
            if prompt:
                # tools += [MyFinalAnswerTool(), DisplayAnswerTool()]
                tools += [DisplayAnswerTool()]
                tools += mcp_tools
                agent = CodeAgent(
                    tools=tools,
                    model=llm,
                    additional_authorized_imports=AUTHORIZED_IMPORTS,
                    max_steps=5,  # for debug
                )
                with strecorder:
                    stream_to_streamlit(
                        agent, PRE_PROMPT + prompt, additional_args={"st": answer_widget}, display_details=False
                    )
                scroll_to_here()

            #        debug(st.session_state.agent_output)
            with answer_widget:
                update_display()
    # use your tools here.
    finally:
        if mcp_client:
            mcp_client.disconnect()
