import asyncio

from dotenv import load_dotenv

load_dotenv()
from browser_use import Agent
from browser_use.llm import ChatOpenAI


async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="o4-mini", temperature=1.0),
    )
    await agent.run()


asyncio.run(main())
