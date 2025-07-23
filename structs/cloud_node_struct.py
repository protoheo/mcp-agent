from langchain_core.runnables import Runnable
from typing import Any


class CloudLLMNode(Runnable):
    def __init__(self, model_manager):
        self.counter = 0
        self.model_manager = model_manager

    def invoke(self,
               state: dict,
               config=None,
               **kwargs: Any) -> dict:

        input_message = state["messages"][-1]["content"]
        print("CloudLLMNode  --->", input_message)
        self.counter += 1
        last_message = f'{input_message}_{str(self.counter).zfill(2)}회 호출'
        wrapped_msg = self.model_manager.msg_wrapper('assistant', last_message)
        state["messages"].append(wrapped_msg)
        print("CloudLLMNode  --->", last_message)

        return {"next": "END", "messages": state["messages"]}
