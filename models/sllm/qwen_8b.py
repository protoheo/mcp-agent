import json
import re
from typing import Any

import torch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

model_name = "Qwen/Qwen3-8B"


class Qwen8BChatModel(BaseChatModel):
    # Pydantic model fields
    system_prompt: str = Field(default="")
    bound_tools: list[BaseTool] = Field(default_factory=list)

    # Model and tokenizer as Pydantic fields
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)

    # Model configuration
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, system_prompt: str = "", **kwargs):
        # Set default values
        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = system_prompt or self._default_system_message()
        if "bound_tools" not in kwargs:
            kwargs["bound_tools"] = []

        super().__init__(**kwargs)

        # Initialize model and tokenizer
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface_causal_model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        enable_thinking = kwargs.pop("enable_thinking", False) if kwargs else False

        if not self._contains_system_message(messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages

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
        raw_content = self.tokenizer.decode(outputs, skip_special_tokens=True).strip(
            "\n"
        )

        # Parse thinking content
        try:
            # Find last occurrence of 151668 (</think>)
            index = len(outputs) - outputs[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            outputs[:index], skip_special_tokens=True
        ).strip("\n")
        raw_content = self.tokenizer.decode(
            outputs[index:], skip_special_tokens=True
        ).strip("\n")

        # Parse tool calls
        tool_calls = self._parse_tool_calls(raw_content)
        content = json.dumps(self.find_json_in_content(raw_content))
        # Create AIMessage
        if tool_calls:
            message = AIMessage(content=content, tool_calls=tool_calls)
        else:
            message = AIMessage(content=content)

        generation = ChatGeneration(message=message)
        llm_output = {"thinking_content": thinking_content}
        return ChatResult(generations=[generation], llm_output=llm_output)

    def _parse_tool_calls(self, content: str) -> list[ToolCall]:
        """Parse tool calls from generated text"""
        tool_calls = []

        # Parse function calls in Qwen model format
        function_call_pattern = r"<function_call>(.*?)</function_call>"
        matches = re.findall(function_call_pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match.strip())
                tool_call = ToolCall(
                    name=call_data.get("name"),
                    args=call_data.get("arguments", {}),
                    id=f"call_{i}",
                )
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

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
                content = message.content
            elif isinstance(message, HumanMessage):
                role = "user"
                content = message.content
            elif isinstance(message, AIMessage):
                role = "assistant"
                content = message.content
                # Convert to function call format if tool_calls exist
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        content += f"\n<function_call>\n{json.dumps({'name': tool_call['name'], 'arguments': tool_call['args']})}\n</function_call>"
            elif isinstance(message, ToolMessage):
                role = "tool"
                content = f"Function result: {message.content}"
            else:
                role = "user"
                content = str(message.content)

            result.append({"role": role, "content": content})
        return result

    def _contains_system_message(self, messages):
        return any(isinstance(m, SystemMessage) for m in messages)

    def _default_system_message(self):
        return "You are a helpful assistant."

    def bind_tools(self, tools, **kwargs) -> "Qwen8BChatModel":
        """LangChain standard tool binding method"""
        # Include tool schema in system prompt
        tool_schemas = self._format_tools_for_prompt(tools)
        enhanced_system_prompt = f"{self.system_prompt}\n\n{tool_schemas}"

        new_model = self.__class__(enhanced_system_prompt)
        new_model.bound_tools = tools
        # Copy existing model and tokenizer to new instance
        object.__setattr__(new_model, "tokenizer", self.tokenizer)
        object.__setattr__(new_model, "model", self.model)
        return new_model

    def find_json_in_content(self, content: str, required_keys: list = None) -> dict:
        """
        Extract JSON data from a string content.

        Args:
            content (str): The string content to search for JSON data.
            required_keys (list, optional): List of required keys to validate in JSON.

        Returns:
            dict: Parsed JSON data if found, otherwise an empty dictionary.
        """
        open_brace_exist = content.count("{") > 0
        close_brace_exist = content.count("}") > 0

        json_str = None

        if open_brace_exist and close_brace_exist:  # Check if both braces exist
            json_match = re.search(r"\{.*?\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group()

        elif open_brace_exist and not close_brace_exist:  # Only open brace exists
            # Find from first { to end of content and try to add }
            start_idx = content.find("{")
            incomplete_json = content[start_idx:]
            json_str = incomplete_json + "}"

        elif not open_brace_exist and close_brace_exist:  # Only close brace exists
            # Find from start to last } and try to add {
            end_idx = content.rfind("}") + 1
            incomplete_json = content[:end_idx]
            json_str = "{" + incomplete_json

        if json_str:
            try:
                parsed_json = json.loads(json_str)

                # Validate required keys if provided
                if required_keys:
                    missing_keys = [
                        key for key in required_keys if key not in parsed_json
                    ]
                    if missing_keys:
                        print(f"Warning: Missing required keys: {missing_keys}")

                return parsed_json
            except json.JSONDecodeError:
                return {}

        return {}

    def _format_tools_for_prompt(self, tools) -> str:
        """Convert tools to prompt format"""
        if not tools:
            return ""

        tool_descriptions = []
        for tool in tools:
            schema = (
                tool.args_schema.schema()
                if hasattr(tool, "args_schema") and tool.args_schema
                else {}
            )
            tool_desc = f"""
    Function: {tool.name}
    Description: {tool.description}
    Parameters: {json.dumps(schema.get('properties', {}), indent=2)}
    """
            tool_descriptions.append(tool_desc)

        return f"""Available Functions:
    {'\n'.join(tool_descriptions)}

    When you need to call a function, use this format:
    <function_call>
    {{"name": "function_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
    </function_call>
    """
