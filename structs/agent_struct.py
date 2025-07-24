from langgraph.graph import StateGraph, END

from structs.rounter_struct import MainRouter
from structs.sllm_node_struct import LLMNode
from structs.cloud_node_struct import CloudLLMNode


class AgenticModel:
    def __init__(self, model_manager):
        self.model_manager = model_manager

        self.builder = StateGraph(dict)
        self.graph = None

        self.do_build()
        self.messages = []

    def do_build(self):
        self.builder.add_node("Router", MainRouter(self.model_manager))
        self.builder.add_node("sLLM", LLMNode(self.model_manager))
        self.builder.add_node("CloudLLM", CloudLLMNode(self.model_manager))
        # self.builder.add_node("END", END)  # (optional, could be dummy)

        # 흐름 정의
        self.builder.set_entry_point("Router")

        self.builder.add_conditional_edges(
            "Router",
            lambda state: state["next"],
            {
                "sLLM": "sLLM",
                "CloudLLM": "CloudLLM",
                "END": END
            }
        )

        self.builder.add_edge("Router", END)
        self.builder.add_edge("sLLM", END)
        self.builder.add_edge("CloudLLM", END)

        self.graph = self.builder.compile()

    def run_chat(self, prompt):
        user_msg = self.model_manager.msg_wrapper('user', prompt)

        self.messages.append(user_msg)

        state = {"messages": self.messages}
        result = self.graph.invoke(state)

        self.messages = result['messages']
        print("run_chat result")
        for msg in result['messages']:
            print("\t", msg)
        # print("Assistant:", result["messages"][-1]["content"])
        # ret = {"Assistant:", result["messages"][-1]["content"]}
        return result['messages']

