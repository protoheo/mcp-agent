from langchain_core.runnables import Runnable
from typing import Any


class LLMNode(Runnable):
    def __init__(self, model_manager):
        self.counter = 0
        self.model_manager = model_manager

    def invoke(self,
               state: dict,
               config=None,
               **kwargs: Any) -> dict:
        input_messages = state["messages"]
        print("sLLMNode  --->", input_messages)

        output_message = self.model_manager.run(input_messages, shot_mode=False).lstrip()
        wrapped_msg = self.model_manager.msg_wrapper('assistant', output_message)
        state["messages"].append(wrapped_msg)
        print("sLLMNode  --->", output_message)

        return {"next": "END", "messages": state["messages"]}
