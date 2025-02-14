import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from langchain_openai import ChatOpenAI

# "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
browser_path = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
browser = Browser(
    config=BrowserConfig(
        headless=False,
        # NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
        chrome_instance_path=browser_path,
    )
)
controller = Controller()


async def main() -> None:
    task = "In docs.google.com write my Papa a quick thank you for everything letter \n - Magnus"
    task += " and save the document as pdf"
    model = ChatOpenAI(model="gpt-4o")
    agent = Agent(
        task=task,
        llm=model,
        controller=controller,
        browser=browser,
    )

    await agent.run()
    await browser.close()

    input("Press Enter to close...")


# if __name__ == "__main__":
#     asyncio.run(main())


async def main() -> None:
    agent = Agent(
        task="Go to Reddit, search for 'browser-use' in the search bar, click on the first post and return the first comment.",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
