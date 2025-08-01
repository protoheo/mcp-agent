from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import torch
import configs.config_prompt as cp


class ModelManager:
    def __init__(self, tokenizer=None, model=None):
        self.tokenizer = tokenizer
        self.model = model

        if self.model is None:
            self.load_model()

        # self.load_model()
    def load_model(self):
        model_name = "Qwen/Qwen1.5-7B-Chat"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def run(self, input_prompts, shot_mode):
        tokenizer = self.tokenizer
        model = self.model
        messages = []
        if shot_mode:
            sys_prompt = {"role": "system",
                          "content": cp.SHOT_PROMPT}
            messages.append(sys_prompt)
        else:
            sys_prompt = {"role": "system",
                          "content": cp.BASE_PROMPT}
            messages.append(sys_prompt)

        messages.extend(input_prompts)
        print("[model_manager]", messages)

        # print("Run model!")

        prompt_ids = tokenizer.apply_chat_template(messages,
                                                   tokenizer=False,
                                                   return_tensors="pt",
                                                   add_generation_prompt=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_ids.to(model.device),
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.001,
                eos_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.decode(outputs[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
        return result

    def msg_wrapper(self, side, prompt):
        return {"role": f"{side}", "content": str(prompt)}
