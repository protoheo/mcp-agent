from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# 🔹 Qwen 로드
model_name = "Qwen/Qwen1.5-4B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
model.eval()


def wrap_message(user_prompt, shot_mode):

    messages = []
    user_prompt = {"role": "user", "content": user_prompt}
    if shot_mode:
        sys_prompt = {"role": "system",
                      "content": "You are an intelligent AI assistant that determines whether the user's request is related to insurance business.\n\nRespond only with:\n- True → if the task is related to insurance (e.g., claims, premiums, underwriting, policies, etc.)\n- False → if it is unrelated.\n\nAnswer only with `True` or `False`, without any explanation."}

        messages.append(sys_prompt)
    messages.append(user_prompt)

    return messages


def run_cc(prompt):
    messages = wrap_message(prompt)
    # print("Run model!")
    prompt_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=prompt_ids.to(model.device),
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
            eos_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
    return result


if __name__ == '__main__':
    while True:
        prompt = input("chat:")
        if prompt == 'ee':
            break
        ret = run_cc(prompt)
        print(ret.strip())
