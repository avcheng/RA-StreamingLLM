# RA-StreamingLLM: Supporting Longer Contexts on StreamingLLM with RAG

[[slides and video demo](https://docs.google.com/presentation/d/11l45TtzJ13GgEdidc2O64d6VlWNC3LL_XJCuezFwIZ4/edit?usp=sharing)]

This fork of StreamingLLM extends the framework to remember context tokens that have been evicted from its current context window. The end goal is enabling LLMs to have "longer-term memory" by remembering older pieces of information in a long-standing conversation. We achieve this by using a retrieval-augmented generation (RAG) framework to store evicted tokens in an additional global database that can then be queried during inference.

## Abstract
StreamingLLM was created as an efficient framework to enables large language models (LLMs) trained with a finite length attention window to generalize to infinite sequence lengths without fine-tuning. However, one limitation of StreamingLLM is its inability to consider previously evicted tokens. The result is a potential memory lapse of items outside of the current window. In this paper, we use concepts from Retrieval Augmented Generation (RAG) to enable infinite-length content storage to improve StreamingLLM's memory. Through our ad hoc testing, we determine that our RA-StreamingLLM maintains infinite window processing and coherence while enhancing accuracy and relevance relative to StreamingLLM when a user inputs long prompts or a series of prompts. 

## Usage

### Environment Setup

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
pip install chromadb redis

python setup.py develop
```

### Run Streaming Llama Chatbot

Initialize the Redis DB on port 6379. Then, run the following command to start the chatbot:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py  --enable_streaming --use_retrieval
```
