# RAG Project Using a Hugging Face Open Source LLM

This project is a simple demonstration of a **Retrieval-Augmented Generation (RAG)** system that combines document retrieval with language generation using open source models from Hugging Face. It uses LangChain to integrate document retrieval (via FAISS and Sentence Transformers) with text generation (using `distilgpt2` from Hugging Face).

## Features

- **Retrieval:** Uses FAISS to store and search document embeddings.
- **Embeddings:** Generates document embeddings using the Sentence Transformer model `all-MiniLM-L6-v2`.
- **Generation:** Uses the open source `distilgpt2` model for text generation.
- **Integration:** Combines retrieval and generation in a seamless QA pipeline using LangChain.

## Requirements

- Python 3.7 or higher

### Python Packages

- `langchain`
- `transformers`
- `faiss-cpu`
- `sentence-transformers`

You can install all the required packages using pip:


pip install langchain transformers faiss-cpu sentence-transformers
