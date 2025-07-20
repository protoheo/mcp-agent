from langgraph.graph import StateGraph, END

from typing import TypedDict

class MyState(TypedDict):
    node: str
    messages: list[dict]


builder = StateGraph(dict)
