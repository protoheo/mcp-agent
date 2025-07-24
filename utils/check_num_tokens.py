from transformers import AutoTokenizer


def check_num_tokens(model_name: str, text: str, include_special_token=True) -> int:
    """
    Check the number of tokens in the given text using the specified model's tokenizer.

    Args:
        model_name (str): The name of the model to use for tokenization.
        text (str): The input text to tokenize.

    Returns:
        int: The number of tokens in the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if include_special_token:
        text = f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
    tokens = tokenizer(text, return_tensors="pt")
    return tokens.input_ids.shape[1]
