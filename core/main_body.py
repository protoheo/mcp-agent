from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# 🔹 Qwen 로드
model_name = "Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
model.eval()

# 🔹 간단한 툴 정의 (예: 계산기)
def calculator_tool(input_str):
    try:
        result = eval(input_str)
        return f"계산 결과는 {result}입니다."
    except Exception as e:
        return f"계산 오류: {e}"

# 🔹 라우터: 툴 호출 여부 판단
class ToolRouter(Runnable):
    def invoke(self, state, config=None):
        last_message = state["messages"][-1]["content"]
        if "계산해줘" in last_message or any(op in last_message for op in ["+", "-", "*", "/"]):
            return {"next": "Tool", "messages": state["messages"]}
        else:
            return {"next": "Qwen", "messages": state["messages"]}

# 🔹 툴 실행 노드
class ToolNode(Runnable):
    def invoke(self, state, config=None):
        input_str = state["messages"][-1]["content"]
        tool_result = calculator_tool(input_str)
        return {"messages": state["messages"] + [{"role": "assistant", "content": tool_result}]}

# 🔹 Qwen 응답 노드
class QwenNode(Runnable):
    def invoke(self, state, config=None):
        prompt = state["messages"][-1]["content"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return {"messages": state["messages"] + [{"role": "assistant", "content": result.strip()}]}

# 🔹 상태 머신 구성
builder = StateGraph(dict)

builder.add_node("Router", ToolRouter())
builder.add_node("Tool", ToolNode())
builder.add_node("Qwen", QwenNode())

# 흐름 정의
builder.set_entry_point("Router")
builder.add_conditional_edges(
    "Router",
    lambda state: state["next"],
    {
        "Tool": "Tool",
        "Qwen": "Qwen"
    }
)
builder.set_finish_point("Tool")
builder.set_finish_point("Qwen")

graph = builder.compile()

# 🔹 테스트
def run_chat(prompt):
    print(f"User: {prompt}")
    state = {"messages": [{"role": "user", "content": prompt}]}
    result = graph.invoke(state)
    print("Assistant:", result["messages"][-1]["content"])

# ✅ 예제 실행
run_chat("2 + 3")
run_chat("what is the capital of France?")
