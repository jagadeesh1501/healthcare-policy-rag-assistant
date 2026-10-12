# Healthcare Policy RAG — V1 to V4

Educational RAG project using the supplied Bajaj Allianz Global Health Care policy wording and supporting PDFs. Not an official insurer service and does not guarantee coverage or claim approval.

## Files
- `v1_langchain_basics.py`: prompt → Google Gemini → output parser using LCEL; no retrieval.
- `v2_faiss_rag.py`: PDF loading, chunking, Hugging Face embeddings, FAISS similarity search.
- `v3_chroma_rag.py`: same pipeline with ChromaDB and source/page/document-type metadata.
- `v4_reranking.py`: FAISS retrieves up to 15 chunks; CrossEncoder reranks and passes top 4 to the LLM.
- `app.py`: Streamlit web app using V4 with Google Gemini.
- `evaluate.py`: runs the same 20 questions against V2, V3 and V4 and writes timings, answers and source pages to CSV.

## Setup (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Create a Gemini API key at https://aistudio.google.com/apikey. Copy `.env.example` to `.env`, then set `GOOGLE_API_KEY` to your key and keep it private. The default chat model is configured by `GEMINI_MODEL` (currently `gemini-2.5-flash`); use a model available to your API key if this model is unavailable. Place policy/supporting PDFs in `data/`. The loader reads all PDFs in that folder. First run downloads the embedding and reranker models, so it may take a while. The chat model uses Gemini API access; embeddings and reranking remain local Hugging Face models.

## Run
```bash
python v1_langchain_basics.py
python v2_faiss_rag.py
python v3_chroma_rag.py
python v4_reranking.py
streamlit run app.py
python evaluate.py
```

## LangSmith tracing
Set these in `.env`, then restart the process (keep your Gemini key and LangSmith key private):
```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=healthcare-policy-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```
Get the key from https://smith.langchain.com/. Never commit `.env` or publish the key. When tracing is enabled, LangChain traces should appear in the named project.

## Evaluation caveat
The script records runtime, answers, retrieved source pages, and errors. It does not automatically prove correctness. Manually grade answer accuracy and groundedness against the expected answers in `evaluation_questions.json` for a defensible comparison. Page numbers displayed are one-based PDF pages; internal metadata is zero-based.
