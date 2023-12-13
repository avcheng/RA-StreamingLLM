from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.utils import load, download_url, load_jsonl
from tqdm import tqdm
import sys
import re
import time
import os
import json
import argparse
import torch
import warnings


@torch.no_grad()
def start_generate(model, input_ids, past_key_values):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    embeddings = outputs.hidden_states[0]
    return past_key_values, pred_token_idx, embeddings


@torch.no_grad()
def greedy_generate(model, tokenizer, past_key_values, pred_token_idx, embeddings, max_gen_len):
    pos = 0
    generated_ids = [pred_token_idx.item()]
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        embeddings = torch.cat([embeddings, outputs.hidden_states[0]], dim=1)
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values, embeddings


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    past_embeddings = None
    while True:
        prompt = input("Your prompt ('quit' to exit): ")
        if prompt == "quit":
            if kv_cache is not None:
                kv_cache.delete_index()
            break
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n\nASSISTANT: ", end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space_db(
                past_key_values, past_embeddings, space_needed)

        past_key_values, pred_token_idx, embeddings = start_generate(
            model, input_ids, past_key_values)

        past_key_values = kv_cache.add_relevant_kv_to_cache(
                past_key_values, embeddings, 5)

        past_key_values, past_embeddings = greedy_generate(
            model, tokenizer, past_key_values, pred_token_idx, embeddings, max_gen_len=max_gen_len
        )


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size, use_retrieval=args.use_retrieval
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    parser.add_argument("--use_retrieval", action="store_true")
    args = parser.parse_args()

    main(args)
