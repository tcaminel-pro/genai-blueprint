import webbrowser
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.ai_core.llm_factory import get_llm
from src.ai_core.mcp_client import get_mcp_servers_dict
from src.utils.config_mngr import global_config
from src.utils.langgraph import print_astream


async def run_smollagent_shell(
    llm_id: str | None, tools: list[BaseTool] = [], mcp_server_names: list[str] = []
) -> None:
    """Run an interactive shell for sending prompts to SmolAgents agents.

...