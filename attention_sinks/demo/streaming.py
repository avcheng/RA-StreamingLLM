import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils import FileStreamer

import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    set_global_tokenizer
)



def create_prompts(samples: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {"prompt": [prompt for prompts in samples["prompt"] for prompt in prompts]}


@torch.no_grad()
def greedy_generate(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: Dataset, log_file: str, max_new_tokens: int = 1000
):
    streamer = FileStreamer(tokenizer, log_file)
    past_key_values = None
    new_line_tokens = tokenizer("\n\n", return_tensors="pt", add_special_tokens=False).input_ids
    service_context = ServiceContext.from_defaults(llm=model, embed_model='local')

    if not os.path.exists("./storage"):
        # load the documents and create the index
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist()
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        
    query_engine = index.as_query_engine(service_context=service_context)

    # storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # index = load_index_from_storage(storage_context)

    while True:

        prompt = input("Enter prompt ('quit' to exit): ")
        if prompt == 'quit':
            break
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        streamer.put(input_ids)

        for _ in range(max_new_tokens):
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            streamer.put(pred_token_idx)
            input_ids = pred_token_idx

            if pred_token_idx == tokenizer.eos_token_id:
                break

        streamer.put(new_line_tokens)


def main():
    parser = argparse.ArgumentParser()

    # Which experiment to run?
    parser.add_argument(
        "--experiment", choices=["attention_sinks", "transformers", "windowed"], default="attention_sinks"
    )

    # Model args
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", action="store_true")

    # Dataset args, not recommended to change:
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/mt_bench_prompts")

    # Where to log
    parser.add_argument("--log_file", type=str, default=None)

    # Window size for windowed and attention_sinks
    parser.add_argument("--window_size", type=int, default=1024)

    # Attention Sinks-only settings
    # Attention Sink window size is calculated with args.window_size - args.attention_sink_size
    parser.add_argument("--attention_sink_size", type=int, default=4)

    args = parser.parse_args()

    # Initialize the model, either via transformers or via attention_sinks
    if args.experiment == "transformers":
        from transformers import AutoModelForCausalLM
    else:
        from attention_sinks import AutoModelForCausalLM
    kwargs = {}
    if args.experiment == "attention_sinks":
        kwargs = {
            "attention_sink_size": args.attention_sink_size,
            "attention_sink_window_size": args.window_size - args.attention_sink_size,  # default: 1020
        }
    elif args.experiment == "windowed":
        kwargs = {
            "attention_sink_size": 0,
            "attention_sink_window_size": args.window_size,
        }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        **kwargs,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))
    tokenizer.pad_token_id = tokenizer.eos_token_id

    set_global_tokenizer(
        tokenizer.encode
    )

    # Set up the dataset
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(create_prompts, batched=True, remove_columns=dataset.column_names)

    log_file = args.log_file or Path("demo") / "streaming_logs" / args.experiment / f"{args.model_name_or_path}.txt"
    greedy_generate(model, tokenizer, dataset, log_file=log_file)


if __name__ == "__main__":
    main()
