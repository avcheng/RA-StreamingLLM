import warnings

warnings.filterwarnings("ignore")

import torch
import argparse

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from llama_index import (
    VectorStoreIndex, 
    StorageContext, 
    ServiceContext, 
    Document, 
    load_index_from_storage,
    set_global_service_context
)
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.node_parser import SimpleNodeParser


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    output = ""
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
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
            output += " ".join(generated_text[pos:now])
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    output += " ".join(generated_text[pos:])
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values, output


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, service_context=None, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)

    try:
        # Rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # Try to load the index from storage
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        # If index not found, create a new one
        index = VectorStoreIndex.from_documents([Document(text="")])
        # Persist index to disk
        index.storage_context.persist()

    while True:
        prompt = input("Your input: ")

        print("\n\nASSISTANT: ", end="")

        new_document = Document(text=prompt)
        nodes = parser.get_nodes_from_documents([new_document])

        # Add to llama index
        index.insert_nodes(nodes)

        # Get context from llama index
        query_engine = index.as_query_engine(response_mode="compact", service_context=service_context, verbose=False)
        context = query_engine.query(prompt)
        # print(context)


        prompt = "USER: Answer the prompt: " + prompt + ". Using the following context: " + str(context) + "\n\nASSISTANT: "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values, output = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )

        new_document = Document(text=output)
        nodes = parser.get_nodes_from_documents([new_document])

        # Add to llama index
        index.insert_nodes(nodes)
        print("Assistant finished.")


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    query_wrapper_prompt = PromptTemplate(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query_str}\n\n### Response:"
    )

    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.25, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="Writer/camel-5b-hf",
        model_name="Writer/camel-5b-hf",
        device_map="auto",
        tokenizer_kwargs={"max_length": 2048},
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16, "offload_folder": "li_offload"},
        model_kwargs={"offload_folder": "li_offload"},
    )
    service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm, embed_model='local')
    set_global_service_context(service_context)

    streaming_inference(
        model,
        tokenizer,
        None,
        service_context,
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
    args = parser.parse_args()

    main(args)
