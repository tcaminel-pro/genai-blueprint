import asyncio

from browser_use import Agent
from dotenv import load_dotenv

from src.ai_core.llm import get_llm

load_dotenv()

llm = get_llm(llm_id="gpt_4o_azure")


async def main():
    agent = Agent(task="Compare the price of gpt-4o and DeepSeek-V3", llm=llm)
    await agent.run()


asyncio.run(main())
