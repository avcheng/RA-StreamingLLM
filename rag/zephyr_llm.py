import torch
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from attention_sinks import AutoModelForCausalLM
import os.path

from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

import chromadb



model_id = "HuggingFaceH4/zephyr-7b-beta"

# Load the chosen model and corresponding tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # for efficiency:
    device_map="auto",
    torch_dtype=torch.float16,
    # `attention_sinks`-specific arguments:
    attention_sink_size=4,
    attention_sink_window_size=252, # <- Low for the sake of faster generation
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id


# initialize client
db = chromadb.PersistentClient(path="./chroma_db")

# get collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

while True:
    # Get user input
    text = input("Enter your text (or type 'quit' to exit): ")
    
    # Check if the user wants to quit
    if text.lower() == 'quit':
        break

    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # A TextStreamer prints tokens as they're being generated
        streamer = TextStreamer(tokenizer)
        generated_tokens = model.generate(
            input_ids,
            generation_config=GenerationConfig(
                # use_cache=True is required, the rest can be changed up.
                use_cache=True,
                min_new_tokens=100_000,
                max_new_tokens=1_000_000,
                penalty_alpha=0.6,
                top_k=5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ),
            streamer=streamer,
        )
        # Decode the final generated text
        output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        print(output_text)