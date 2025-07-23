from typing import TypedDict

import dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

_ = dotenv.load_dotenv()


class AgentState(TypedDict):
    """State for the agent."""

    name: str
    age: int


def is_proper_request(): ...


def node_sllm(): ...


def node_llm(): ...


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("sllm", node_sllm)
    graph.add_node("llm", node_llm)

    graph.add_edge(START, "sllm")
    graph.add_conditional_edges("sllm", is_proper_request, "llm", END)

    graph.compile()
    return graph


def main():
    graph = build_graph()
