import random
from typing import TypedDict

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from models import Qwen8BChatModel

_ = dotenv.load_dotenv()


class AgentState(TypedDict):
    """State for the agent."""

    messages: list
    revision: int
    max_revisions: int


def init():
    global sllm, llm
    with open("prompts/system/sllm.md", "r") as f:
        sllm_system_prompt = f.read()
    sllm_system_prompt = sllm_system_prompt.replace("<|Tool|>", "capital_city_finder")
    sllm = Qwen8BChatModel(sllm_system_prompt)
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )


def is_proper_request():
    """Check if the request is proper."""
    return random.choice(["llm", END])


def node_sllm(state: AgentState):
    """Node for the SLLM."""
    global sllm
    messages = state.get("messages", [])
    result = sllm.invoke(messages)
    # 결과를 state에 추가
    new_messages = messages + [result.content]
    return {"messages": new_messages}


def node_llm(state: AgentState):
    """Node for the LLM."""
    global llm
    messages = state.get("messages", [])
    result = llm.invoke(messages)
    # 결과를 state에 추가
    new_messages = messages + [result]
    return {"messages": new_messages}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("sllm", node_sllm)
    builder.add_node("llm", node_llm)

    builder.add_edge(START, "sllm")
    builder.add_conditional_edges("sllm", is_proper_request, {"llm": "llm", END: END})

    graph = builder.compile()
    return graph


def main():
    graph = build_graph()
    for chunk in graph.stream(
        AgentState(
            messages=[HumanMessage(content="안녕, 한국의 수도는 어디야?")],
            revision=0,
            max_revisions=5,
        )
    ):
        for event in chunk.values():
            print(event["messages"][-1].content)


if __name__ == "__main__":
    init()
    main()
