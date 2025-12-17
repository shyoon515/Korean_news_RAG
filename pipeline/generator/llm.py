from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from tqdm import tqdm

def load_llm(model_name: str, cache_dir: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    """
    Load the language model and tokenizer.
    Args:
        model_name (str): The name of the model to load.
    Returns:
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    if model_name == 'llama':
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map='auto',
        cache_dir=cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    return model, tokenizer

def load_lora_model(model_name: str, special_tokens : List[str], lora_config: LoraConfig, cache_dir: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a LoRA model with the specified configuration.
    Args:
        model_name (str): The name of the base model to load.
        special_tokens (List[str]): List of special tokens to add to the tokenizer. -> retrieval selection, think retrieval, think qr special tokens
        lora_config (LoraConfig): The configuration for LoRA.
        cache_dir (str): Directory to cache the model files.
    Returns:
        model (AutoModelForCausalLM): The loaded LoRA model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
    """
    model, tokenizer = load_llm(model_name, cache_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.resize_token_embeddings(len(tokenizer))
    lora_model = get_peft_model(model, lora_config)
    lora_model.config.label2id = {token: idx for idx, token in enumerate(special_tokens)}
    lora_model.config.id2label = {idx: token for token, idx in lora_model.config.label2id.items()}
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return lora_model, tokenizer

def load_lora_classifier(model_name: str, classifier_tokens, lora_config, cache_dir: str = None):
    if model_name == 'llama':
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(classifier_tokens),
        torch_dtype="auto",
        device_map='auto',
        cache_dir=cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": classifier_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    clf_model = get_peft_model(model, lora_config)
    clf_model.config.label2id = {token: idx for idx, token in enumerate(classifier_tokens)}
    clf_model.config.id2label = {idx: token for token, idx in clf_model.config.label2id.items()}
    return clf_model, tokenizer

def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    formatted_prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 128,
    sample: bool = False,
    logger=None
) -> List[str]:
    """
    Generate outputs in batches using the model and tokenizer.
    Args:
        model (AutoModelForCausalLM): The language model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        formatted_prompts (List[str]): List of formatted prompts to generate from.
        batch_size (int): Number of prompts to process in parallel.
    """
    
    tokenizer.padding_side = 'left' # decoder-only models : Llama, OPT, MistralS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_decoded_outputs = []
    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating outputs"):
        batch_prompts = formatted_prompts[i:i + batch_size]

        all_messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            for prompt in batch_prompts
        ]
        input_ids = tokenizer.apply_chat_template(
            all_messages,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        if sample:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True
            )
        else:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=1e-10,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True
            )

        sequences = outputs.sequences
        generated_ids = [
            seq[input_id.shape[0]:] for input_id, seq in zip(input_ids, sequences)
        ]

        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_decoded_outputs.extend(decoded_outputs)

        if logger:
            generated_pairs = [(prompt, response) for prompt, response in zip(batch_prompts, decoded_outputs)]
            for prompt, response in generated_pairs:
                logger.info(f"Prompt: {prompt}\nResponse: {response}")
    return all_decoded_outputs