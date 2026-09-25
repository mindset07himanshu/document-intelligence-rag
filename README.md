# document-intelligence-rag

RAG-based Document Intelligence Copilot for context-aware document question answering using LLMs and vector search.

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using semantic search and an LLM.

The system:
- Loads PDF documents from the `data/` directory
- Splits documents into overlapping chunks
- Generates embeddings for semantic retrieval
- Stores vectors using Chroma
- Retrieves relevant context for user questions
- Generates context-aware answers using an LLM

## Project Structure

```text
document-intelligence-rag/
├── data/
│   ├── SN_P1_eng_2021.pdf
│   ├── insurance_knowledge_base.pdf
│   └── SN_MPF_Eng.pdf
├── eval/
│   ├── benchmark.py
│   └── questions.json
├── src/
├── tests/
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
