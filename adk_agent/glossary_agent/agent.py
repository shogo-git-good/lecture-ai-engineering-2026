
from google.adk.agents.llm_agent import Agent

# 演習で作ったPythonツール
from app.tools import (
    search_glossary,
    get_lecture_summary,
    make_quiz,
)

# 演習で使った指示文
from app.agent import INSTRUCTIONS

# Google ADKでエージェントを定義
root_agent = Agent(
    model="gemini-3.5-flash-lite",
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[
        search_glossary,
        get_lecture_summary,
        make_quiz,
    ],
)
