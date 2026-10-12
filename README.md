# 🏥 Healthcare Policy RAG Assistant
### From Basic Retrieval to Cross-Encoder Reranking — V1 to V4

An AI-powered **Healthcare Policy Question Answering Assistant** designed to help users explore health insurance documents through natural-language questions. Built with LangChain, Google Gemini, Hugging Face embeddings, vector databases, and Cross-Encoder reranking.

> ⚠️ **Disclaimer:** This is an educational project built around the supplied Bajaj Allianz Global Health Care policy wording and supporting PDFs. It is not an official insurer service and does not guarantee policy coverage or claim approval. Always verify answers against the applicable policy documents.

---

## ✨ What Makes This Project Interesting?

Insurance policies can be lengthy and difficult to navigate. This project explores how Retrieval-Augmented Generation (RAG) can make policy information easier to find while retaining references to supporting documents.

Instead of relying only on a language model's general knowledge, the retrieval-based versions search policy documents and provide relevant passages to the model.

## 🚀 Project Evolution: V1 → V4

| Version | Approach | Key Capability |
|---|---|---|
| **V1** | Basic LangChain + LCEL | Prompting, Gemini generation, and output parsing |
| **V2** | FAISS-based RAG | Semantic retrieval from PDF documents |
| **V3** | ChromaDB-based RAG | Vector storage with source and page metadata |
| **V4** | FAISS + Cross-Encoder | Retrieve up to 15 candidates, rerank them, and send the top 4 to the LLM |

**The goal:** Explore how retrieval architecture affects relevance, response time, and the quality of evidence supplied to the language model.

## 🧠 How It Works

1. **Document ingestion** — Load the policy and supporting PDF documents.
2. **Text chunking** — Split documents into smaller passages for retrieval.
3. **Embedding generation** — Convert text chunks into vector representations using local Hugging Face models.
4. **Semantic retrieval** — Find passages relevant to the user's question using FAISS or ChromaDB.
5. **Reranking (V4)** — Use a Cross-Encoder to score retrieved passages and select the most relevant evidence.
6. **Answer generation** — Send the selected context to Google Gemini to generate a response.
7. **Source inspection** — Display source documents and page references to help users verify the answer.

## 🛠️ Tech Stack

- **Language:** Python
- **LLM orchestration:** LangChain, LCEL
- **Language model:** Google Gemini
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector databases:** FAISS, ChromaDB
- **Reranking:** Cross-Encoder
- **User interface:** Streamlit
- **Observability:** LangSmith
- **Evaluation:** Python, CSV, and a 20-question evaluation set

## 📁 Project Structure

```text
healthcare-policy-rag-assistant/
├── data/                       # Policy and supporting PDFs
├── app.py                      # Streamlit application
├── v1_langchain_basics.py      # Basic LCEL pipeline
├── v2_faiss_rag.py             # FAISS retrieval pipeline
├── v3_chroma_rag.py            # ChromaDB retrieval pipeline
├── v4_reranking.py             # Retrieval + Cross-Encoder reranking
├── rag_common.py               # Shared loading, embeddings and RAG utilities
├── evaluate.py                 # V2, V3 and V4 comparison
├── evaluation_questions.json   # Evaluation questions
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jagadeesh1501/healthcare-policy-rag-assistant.git
cd healthcare-policy-rag-assistant
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.11 is recommended for this project.

### 3. Configure your Gemini API key

Create a key at [Google AI Studio](https://aistudio.google.com/apikey).

Copy `.env.example` to `.env` and configure your key and model:

```dotenv
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Use a model available to your API key. Keep `.env` private and never commit API credentials.

### 4. Add the documents

Place the policy and supporting PDFs in the `data/` directory. The document loader reads all PDFs in that folder, so include only documents you are authorized to use.

The first run downloads the embedding and reranker models. These run locally; Gemini is used for answer generation.

## ▶️ Run the Project

Run individual versions from the project root:

```bash
python v1_langchain_basics.py
python v2_faiss_rag.py
python v3_chroma_rag.py
python v4_reranking.py
```

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Run the evaluation script:

```bash
python evaluate.py
```

## 📊 Evaluation & Experimentation

The evaluation script runs the same 20 questions against V2, V3, and V4, recording:

- Execution time
- Generated answers
- Retrieved source pages
- Errors encountered during execution

These measurements support a practical comparison of the retrieval approaches. **Runtime and retrieval results alone do not establish answer correctness.** Manually assess accuracy, groundedness, and source relevance against the expected answers in `evaluation_questions.json`.

Page references shown to users are one-based PDF page numbers; internal metadata uses zero-based page indices.

## 🔍 LangSmith Observability

LangSmith can be used to inspect supported LangChain traces, investigate failures, and understand model-call behavior.

Add the following to your local `.env` file:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=healthcare-policy-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Get your API key from [LangSmith](https://smith.langchain.com/). Restart the application after changing environment variables, and verify that traces appear in the selected project.

Never publish your API keys or `.env` file.

## 🔮 Potential Improvements

- Expand the evaluation dataset and manually grade answer quality.
- Compare retrieval recall, source relevance, latency, and groundedness.
- Add filters for policy sections, document types, and page references.
- Improve handling of exclusions, waiting periods, sub-limits, and eligibility conditions.
- Add stronger tests for unsupported questions and incorrect citations.
- Deploy the Streamlit app to a public hosting platform.

## 👩‍💻 Author

**Sana Jagadeesh**

GitHub: [@jagadeesh1501](https://github.com/jagadeesh1501)

🔗 **Project Repository:** [Healthcare Policy RAG Assistant](https://github.com/jagadeesh1501/healthcare-policy-rag-assistant)

---

*Built to explore document-grounded question answering, retrieval strategies, and evaluation in healthcare insurance information systems.*
