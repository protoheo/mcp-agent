import json
import os
from typing import TypedDict

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from models import Qwen8BChatModel
from tools.samples import CryptoPriceTool

_ = dotenv.load_dotenv()


class AgentState(TypedDict):
    """State for the agent."""

    history: list
    messages: list
    revision: int
    max_revisions: int


def init():
    global sllm, llm, tool_node, llm_system_prompt
    with open("prompts/system/sllm_v2.md", "r") as f:
        sllm_system_prompt = f.read()
    llm_path = os.path.join(os.path.dirname(__file__), "prompts", "system", "llm.md")
    with open(llm_path, "r") as f:
        llm_system_prompt = f.read()

    crypto_price_finder = CryptoPriceTool()
    sllm = Qwen8BChatModel(sllm_system_prompt)
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", verbose=True)
    llm = llm.bind_tools([crypto_price_finder])
    tool_node = ToolNode(tools=[crypto_price_finder])


def model_router(state: AgentState):
    messages = state.get("history", [])
    last_message = messages[-1]
    try:
        json_content = json.loads(last_message.content)
    except json.JSONDecodeError:
        json_content = {}

    is_valid = json_content.get("is_valid", False)
    if is_valid:
        return "llm"
    else:
        return END


def tool_router(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    if tool_calls:
        return "tool"
    else:
        return END


def node_sllm(state: AgentState):
    """Node for the SLLM."""
    global sllm
    history = state.get("history", [])
    messages = state.get("messages", [])
    result = sllm.invoke(messages)

    new_history = history + [result]
    return {"history": new_history, "messages": messages}


def node_llm(state: AgentState):
    """Node for the LLM."""
    global llm
    history = state.get("history", [])
    messages = state.get("messages", [])
    system_message = SystemMessage(content=llm_system_prompt)
    result = llm.invoke([system_message] + messages)

    new_history = history + [result]
    new_messages = messages + [result]

    return {"history": new_history, "messages": new_messages}


def node_tool(state: AgentState):
    """Node for the tool."""
    global tool_node
    history = state.get("history", [])
    messages = state.get("messages", [])
    result = tool_node.invoke(messages)

    new_history = history + result
    new_messages = messages + result

    return {"history": new_history, "messages": new_messages}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("sllm", node_sllm)
    builder.add_node("llm", node_llm)
    builder.add_node("tool", node_tool)

    # Define edges
    builder.add_edge(START, "sllm")
    builder.add_conditional_edges("sllm", model_router, {"llm": "llm", END: END})
    builder.add_conditional_edges("llm", tool_router, {"tool": "tool", END: END})
    builder.add_edge("tool", "llm")
    graph = builder.compile()
    return graph


def main():
    graph = build_graph()

    # 테스트 메시지들
    test_messages = ["비트코인의 현재 가격은 얼마인가요?", "치킨이 먹고 싶어."]

    for test_msg in test_messages:
        print(f"\n🔹 사용자: {test_msg}")
        print("=" * 50)

        for chunk in graph.stream(
            AgentState(
                history=[HumanMessage(content=test_msg)],
                messages=[HumanMessage(content=test_msg)],
                revision=0,
                max_revisions=5,
            )
        ):
            for node_name, event in chunk.items():
                if "history" in event and event["history"]:
                    last_msg = event["history"][-1]
                    print(f"[{node_name}] ", end="")
                    if hasattr(last_msg, "content"):
                        try:
                            data = json.loads(last_msg.content)
                            content = data.get("content", "")
                            print(content.encode("utf-8").decode("unicode_escape"))
                        except json.JSONDecodeError:
                            # If JSON decoding fails, print the content directly
                            print(last_msg.content)

                    elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"Tool calls: {last_msg.tool_calls}")
                    else:
                        print(str(last_msg))
        print()


if __name__ == "__main__":
    init()
    main()
    main()
