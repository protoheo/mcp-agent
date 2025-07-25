from langchain_core.runnables import Runnable
from typing import Any


class MainRouter(Runnable):
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def invoke(self,
               state: dict,
               config=None,
               **kwargs: Any) -> dict:
        # 일단은 모델 넣어야되는데 지금은 그냥 단순 분기 생성

        if len(state["messages"]) > 4:
            print("MainRouter --> max_length!")
            return {"next": "END", "messages": state["messages"]}

        is_shot = self.model_manager.run(state["messages"], shot_mode=True).strip()
        print("MainRouter------->", is_shot)

        if is_shot == "False":
            return {"next": "sLLM", "messages": state["messages"]}
        else:
            return {"next": "CloudLLM", "messages": state["messages"]}

