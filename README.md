# document-intelligence-rag

RAG-based Document Intelligence Copilot for context-aware document question answering using LLMs and vector search.

## Included demo test set

The repository includes 3 synthetic PDF documents in `data/` so the project can be run immediately without relying on private or external documents:

- `SN_P1_eng_2021.pdf`
- `insurance_knowledge_base.pdf`
- `SN_MPF_Eng.pdf`

These files are explicitly synthetic demo content and should not be presented as official insurance or regulatory documents.

## RAG configuration

- Chunk size: `700`
- Chunk overlap: `120`
- Retriever top-k: `5`
- Minimum relevance score: `0.42`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: Chroma

## Measuring performance

Run:

```bash
python -m eval.benchmark
```

The benchmark reports the actual number of PDFs, indexed chunks, retrieval latency, and end


### Local benchmark result

- Test PDFs: 3
- Indexed chunks: 3
- Average retrieval latency: 120.56 ms
- Average total answer latency: 8.89 s

These measurements were obtained using the included synthetic demo documents and may vary depending on hardware, network conditions, model availability, and API response time.

The Streamlit UI also displays the measured response time for each live question.

## Example evaluation questions

The supplied evaluation questions cover risk management, indemnity, and MPF withdrawal conditions. See `eval/questions.json`.
