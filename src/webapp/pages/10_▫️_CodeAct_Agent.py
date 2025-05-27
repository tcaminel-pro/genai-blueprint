"""CodeAct Agent - A Streamlit-based interface for interactive AI-powered code execution.

This module provides a Streamlit web application that allows users to interact with an AI agent
capable of executing Python code in a controlled environment. The agent can perform various tasks
including data analysis, visualization, and web interactions using predefined tools and libraries.

Key Features:
- Interactive chat interface for code execution
- Support for multiple AI models
- Integration with various data sources and APIs
- Safe execution environment with restricted imports
- Real-time output display including plots and maps
"""

from datetime import date
from pathlib import Path
from typing import Any, List

import folium
import pandas as pd
import streamlit as st
import yfinance as yf
from groq import BaseModel
from loguru import logger
from omegaconf import DictConfig
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
from src.utils.config_mngr import global_config
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
    """A tool for working with Pandas DataFrames from various data sources.
    
    This tool provides access to tabular data stored in files and allows the AI agent
    to perform data analysis operations on the loaded DataFrame.
    
    Attributes:
        name: Unique identifier for the tool
        description: Brief description of the tool's functionality
        source_path: Path to the data file containing the DataFrame
    """
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


class CodeactDemo(BaseModel):
    """Configuration class for CodeAct Agent demonstrations.
    
    This class defines the structure for setting up different demo scenarios
    including available tools, MCP servers, and example prompts.
    
    Attributes:
        name: Name of the demonstration
        tools: List of available tools for the demo
        mcp_servers: List of MCP server configurations
        examples: List of example prompts for the demo
    """
    name: str
    tools: list[Tool] = []
    mcp_servers: list[str] = []
    examples: list[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool
def get_stock_info(symbol: str, key: str) -> dict:
    """Retrieve specific information about a stock using its ticker symbol.
    
    This tool interfaces with Yahoo Finance to fetch various stock metrics
    including price, company information, and financial indicators.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL' for Apple)
        key: Specific metric to retrieve from the stock info
        
    Returns:
        Dictionary containing the requested stock information
    """
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
    """Fetch historical stock price data for a given time period.
    
    This tool retrieves daily price data including open, close, high, low prices
    and volume for the specified date range.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for the historical data
        end_date: End date for the historical data
        
    Returns:
        DataFrame containing historical price data with Date and Close price columns
    """
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


# List of authorized Python packages that can be imported in the code execution environment
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


def load_demos_from_config() -> List[CodeactDemo]:
    """Load and configure demonstration scenarios from the application configuration.
    
    This function reads the demo configurations from the global config and creates
    corresponding CodeactDemo objects with their associated tools and examples.
    
    Returns:
        List of configured CodeactDemo objects
    """
    """Load demos configuration from YAML file"""

    tools_dict = {
        "WebSearchTool": WebSearchTool,
        "VisitWebpageTool": VisitWebpageTool,
    }
    try:
        demos_config = global_config().get_list("codeact_agent_demos")
        result = []
        # Create Demo objects from the configuration
        for demo_config in demos_config:
            name = demo_config.get("name", "")
            examples = demo_config.get("examples", [])
            mcp_servers = demo_config.get("mcp_servers", [])

            # Process tools
            tools = []
            for tool_config in demo_config.get("tools", []):
                if isinstance(tool_config, DictConfig):
                    tool_type = tool_config.get("type", "")
                    # Handle different tool types
                    if tool_type == "DataFrameTool":
                        tools.append(
                            DataFrameTool(
                                name=tool_config.get("name", ""),
                                description=tool_config.get("description", ""),
                                source_path=DATA_PATH / tool_config.get("source_path", "").split("/")[-1],
                            )
                        )
                    elif tool_type == "function":
                        func_name = tool_config.get("name", "")
                        if func_name in globals():
                            tools.append(globals()[func_name])
                    elif tool_type in tools_dict:
                        tools.append(tools_dict[tool_type]())
                    else:
                        logger.warning(f"Unknown tool type: {tool_type}")

            demo = CodeactDemo(
                name=name,
                tools=tools,
                mcp_servers=mcp_servers,
                examples=examples,
            )
            result.append(demo)
        return result
    except Exception as e:
        logger.exception(f"Error loading demos from config: {e}")
        return []


# Load demos from config or use defaults
SAMPLES_DEMOS = load_demos_from_config()

##########################
#  UI
##########################

llm_config_widget(st.sidebar)


@st.cache_data(show_spinner=True)
def get_cache_dataframe(file_or_filename: Path | UploadedFile, **kwargs) -> pd.DataFrame:
    """Load and cache a DataFrame from a file or uploaded file object.
    
    This function handles various file formats and caches the loaded DataFrame
    to improve performance for repeated access.
    
    Args:
        file_or_filename: Path to the file or Streamlit UploadedFile object
        **kwargs: Additional arguments to pass to the file reader
        
    Returns:
        Loaded Pandas DataFrame
    """
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
    demo = CodeactDemo(name="custom", examples=[])
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
    """Display the final result of the AI agent's execution.
    
    This tool handles the presentation of different types of results including
    Markdown text, DataFrames, images, and Folium maps in the Streamlit interface.
    
    Args:
        answer: The final result to display, can be various types
        
    Returns:
        String representation of the answer
    """
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
    """Update the Streamlit display with all accumulated agent outputs.
    
    This function iterates through the stored agent outputs and displays
    them in the appropriate format in the Streamlit interface.
    """
    if len(st.session_state.agent_output) > 0:
        st.write("answer:")
    for msg in st.session_state.agent_output:
        display_final_msg(msg)


def display_final_msg(msg: Any) -> None:
    """Display a single message in the appropriate format.
    
    This function handles the rendering of different message types including
    Markdown, DataFrames, images, and Folium maps in the Streamlit interface.
    
    Args:
        msg: The message to display, can be various types
    """
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
