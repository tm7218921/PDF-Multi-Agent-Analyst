
from __future__ import annotations

from agents import Agent

from utils.groq_config import get_groq_model
from utils.prompts import PLANNER_SYSTEM_PROMPT


def build_planner(rag_agent: Agent, summarization_agent: Agent) -> Agent:
    """
    Constructs and returns the Planner Agent.

    Both specialist agents are exposed as tools via agent.as_tool(), which
    wraps each in a function_tool that the Planner can call automatically.
    """
    rag_tool = rag_agent.as_tool(
        tool_name="rag_agent",
        tool_description=(
            "Use this tool to answer specific questions about the content of "
            "uploaded PDF documents. Provide the user's exact question as input."
        ),
    )

    summarization_tool = summarization_agent.as_tool(
        tool_name="summarization_agent",
        tool_description=(
            "Use this tool when the user wants a summary or overview of the "
            "uploaded documents. Optionally include a focus topic."
        ),
    )

    planner = Agent(
        name="Planner",
        instructions=PLANNER_SYSTEM_PROMPT,
        tools=[rag_tool, summarization_tool],
        model=get_groq_model(),
    )

    return planner
