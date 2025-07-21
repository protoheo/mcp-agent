import torch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"


class Qwen8BChatModel(BaseChatModel):
    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,  # Use slow tokenizer for compatibility with chat templates
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface_causal_model"

    # @property
    # def _default_generate_kwargs(self):
    #     return {
    #         "max_new_tokens": 256,
    #         "do_sample": True,
    #         "top_p": 0.9,
    #         "temperature": 0.7,
    #     }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if kwargs:
            enable_thinking = kwargs.pop("enable_thinking", False)
        else:
            enable_thinking = False
        inputs = self.tokenize(messages, enable_thinking=enable_thinking)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
        outputs = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(outputs, skip_special_tokens=True).strip("\n")
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(outputs) - outputs[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            outputs[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            outputs[index:], skip_special_tokens=True
        ).strip("\n")
        result = {"thinking_content": thinking_content, "content": content}
        return result

    def tokenize(self, message, enable_thinking=True):
        converted_messages = self.convert_langgraph_messages(message)
        text = self.tokenizer.apply_chat_template(
            converted_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def convert_langgraph_messages(self, messages):
        result = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, ToolMessage):
                role = "tool"
            else:
                role = f"{message.__name__}"
            result.append({"role": role, "content": message.content})
        return result
