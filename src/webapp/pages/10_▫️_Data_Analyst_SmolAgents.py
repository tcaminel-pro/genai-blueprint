"""AI Chat interface for analyzing data" """

from datetime import date
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import smolagents.default_tools
import streamlit as st
import yfinance as yf
from devtools import debug
from groq import BaseModel
from loguru import logger
from pydantic import ConfigDict
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    Tool,
    VisitWebpageTool,
    tool,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_folium import st_folium

from src.ai_core.llm import LlmFactory
from src.ai_core.prompts import dedent_ws
from src.utils.streamlit.load_data import TABULAR_FILE_FORMATS_READERS, load_tabular_data
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smolagents_streamlit import stream_to_streamlit

MODEL_ID = None  # Use the one by configuration
# MODEL_ID = "qwen_qwq32_deepinfra"
# MODEL_ID = "gpt_o3mini_openrouter"
# MODEL_ID = "qwen_qwq32_openrouter"

DATA_PATH = Path.cwd() / "use_case_data/other"


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
    examples: list[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


SEARCH_TOOLS = [DuckDuckGoSearchTool(), VisitWebpageTool()]


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


DEMOS = [
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
        ],
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
    # Demo(
    #     name="CO2 Emissions",
    #     tools=[
    #         DataFrameTool(
    #             name="titanic data reader",
    #             description="Data related to the Titanic passengers",
    #             source_path=DATA_PATH / "country CO2 emissions",
    #         )
    #     ],
    #     examples=[
    #         "what was the CO2 emissions of Brazil for energy generation in 2023",
    #         "Generate a bar chart with emissions by sector for France in 2022",
    #         "Create a pie chart of emissions by sector",
    #         "Show the change in emissions from the industrial sector between 2022 and 2024",
    #         "Create a simple UI with a multiselect widget and a text ",
    #         "Train a ML model to predict the CO2 evolution of France in the next 2 years. Display the curve with historical and predicted data",
    #     ],
    # ),
    Demo(
        name="Stock Price",
        tools=[get_stock_info, get_historical_price],
        examples=[
            "What is the current price of Meta stock?",
            "Show me the historical prices of Apple vs Microsoft stock over the past 6 months",
        ],
    ),
    Demo(
        name="Geo",
        tools=SEARCH_TOOLS,
        examples=["Display the map of Toulouse"],
    ),
]


@st.cache_data(show_spinner=True)
def get_cache_dataframe(file_or_filename: Path | UploadedFile, **kwargs) -> pd.DataFrame:
    return load_tabular_data(file_or_filename=file_or_filename, **kwargs)


FILE_SElECT_CHOICE = ":open_file_folder: :orange[Select your file]"
selected_pill = st.pills("Demos:", options=[demo.name for demo in DEMOS] + [FILE_SElECT_CHOICE], default=DEMOS[0].name)

raw_data_file = None
df: pd.DataFrame | None = None
tools = []

if selected_pill == FILE_SElECT_CHOICE:
    raw_data_file = st.file_uploader(
        "Upload a Data file:",
        type=list(TABULAR_FILE_FORMATS_READERS.keys()),
        # on_change=clear_submit,
    )
else:
    demo = next(d for d in DEMOS if d.name == selected_pill)
    tools = demo.tools

    col1, col2 = st.columns([3, 1])
    with col2:
        st.write("**Available Tools:**")
        for tool in tools:
            st.write(f"- {tool.name}")

    with col1:
        st.write("**Example Prompts:**")
        st.write(demo.examples)


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

# DOES NOT WORK
smolagents.default_tools.FinalAnswerTool = MyFinalAnswerTool


class DisplayAnswerTool(Tool):
    name = "print_result"
    description = "Display important step in the reasonning (1 sentence) or the final answer to the given query."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        st.session_state.agent_output.append(answer)
        return "answer displayed: {answer}"


def update_display() -> None:
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


st.title("Analytics AI Agent")
llm_config_widget(st.sidebar)

model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)

if "agent_output" not in st.session_state:
    st.session_state.agent_output = []

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
]

FOLIUM_INSTRUCTION = dedent_ws(
    """ 
    - Use Folium to display a map. For example: 
        -- to display map at a given location, call  folium.Map([latitude, longitude])
        -- Do your best to select the zoom factor so whole location enter globaly in the map
        -- output the map object
"""
)

IMAGE_INSTRUCTION = dedent_ws(
    """ 
    -  When creating a plot or generating an image:
      -- save it as png in a tempory directory (use tempfile)
      -- call print_result with the pathlib.Path  
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
    - Call the function 'print_result' to display the final result. It accepts markdown (first choice), str, number, or a pathlib.Path to a generated image, or whenever possible  Python objects of Pandas Dataframe, or Follium Map.
    - Print also the outcome on stdio, or the title if it's a diagram.
    - {FOLIUM_INSTRUCTION}
    - {IMAGE_INSTRUCTION}

    \nRequest :
    """
)


col1, col2 = st.columns(2)
with col1:
    if prompt := st.chat_input("What would you like to ask ?"):
        # st.session_state.agent_output = []
        tools += [DisplayAnswerTool()]
        agent = CodeAgent(
            tools=tools,
            model=llm,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            max_steps=5,  # for debug
        )

        # with st.container(height=600):
        with st.status("Agents thoughts:", expanded=True):
            stream_to_streamlit(agent, PRE_PROMPT + prompt, additional_args={"st": col2}, display_details=False)

    debug(st.session_state.agent_output)
    with col2:
        st.write("answer:")
        update_display()
