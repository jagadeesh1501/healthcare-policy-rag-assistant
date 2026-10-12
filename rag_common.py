
import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Load .env from the same directory as this Python file
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)

PDF_DIR = Path(os.getenv("PDF_DIR", str(BASE_DIR / "data")))
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", str(BASE_DIR / "vectorstore")))

# Resolve relative paths against the project directory
if not PDF_DIR.is_absolute():
    PDF_DIR = BASE_DIR / PDF_DIR

if not VECTOR_DIR.is_absolute():
    VECTOR_DIR = BASE_DIR / VECTOR_DIR

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))


# --------------------------------------------------
# API key validation
# --------------------------------------------------

def get_google_api_key():
    """Read and validate the Gemini API key without exposing it."""

    # Also support an environment variable set directly in the terminal.
    # Reload .env in case it was created or updated after startup.
    load_dotenv(dotenv_path=ENV_FILE, override=False)

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is missing.\n"
            f"Expected .env file at: {ENV_FILE}\n"
            "Add this line to that file:\n"
            "GOOGLE_API_KEY=your_actual_gemini_api_key\n"
            "Then restart Streamlit."
        )

    return api_key


# --------------------------------------------------
# Document processing
# --------------------------------------------------

def infer_document_type(filename):
    name = filename.lower()

    if "claim" in name:
        return "claim procedure guide"
    if "cashless" in name:
        return "cashless hospitalization guide"
    if "schedule" in name or "benefit" in name:
        return "policy schedule / benefit table"
    if "irdai" in name or "consumer" in name:
        return "regulatory / consumer guidance"

    return "policy wording or supporting PDF"


def load_chunks():
    """Load PDFs and split pages into searchable chunks."""

    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DIR.resolve()}"
        )

    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR.resolve()}. "
            "Put policy PDFs in the data/ folder."
        )

    pages = []

    for pdf in pdfs:
        print(f"Loading: {pdf.name}")

        for doc in PyPDFLoader(str(pdf)).load():
            doc.metadata["source"] = pdf.name
            doc.metadata["page"] = int(
                doc.metadata.get("page", 0)
            )
            doc.metadata["document_type"] = infer_document_type(
                pdf.name
            )
            pages.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = [
        doc
        for doc in splitter.split_documents(pages)
        if doc.page_content.strip()
    ]

    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# Cached models
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings():
    """Initialize the local embedding model once per process."""

    print(f"Initializing embedding model: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_llm():
    """Initialize Gemini once per process."""

    api_key = get_google_api_key()

    print(f"Initializing Gemini model: {GEMINI_MODEL}")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
        timeout=60,
        max_retries=1,
    )


# --------------------------------------------------
# RAG chain
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_chain():
    """Build and cache the prompt and Gemini chain."""

    prompt = ChatPromptTemplate.from_template(
        """You are a careful health-insurance policy assistant.

Answer the question using ONLY the supplied context.

Follow these rules:
1. Answer the question directly and concisely.
2. For questions about a policy's purpose, look for statements
   describing what the policy is designed to cover or provide.
3. Distinguish policy wording from general explanations.
4. Do not invent benefits, exclusions, limits, waiting periods,
   premiums, network status, or claim outcomes.
5. Never guarantee that a claim will be approved.
6. If the context does not contain enough evidence, say:
   "I couldn't find sufficient information in the provided documents."
7. Do not claim a source supports something unless its text does.
8. Refer to the source filename and PDF page where relevant.

Question:
{question}

Retrieved policy context:
{context}

Answer:"""
    )

    return prompt | get_llm() | StrOutputParser()


# --------------------------------------------------
# Context and citations
# --------------------------------------------------

def format_context(docs):
    """Format retrieved chunks with source metadata."""

    sections = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 0)) + 1
        doc_type = doc.metadata.get(
            "document_type", "unknown"
        )

        sections.append(
            f"[Source: {source} | PDF page: {page} | "
            f"Type: {doc_type}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def source_rows(docs):
    """Return unique source/page references."""

    rows = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 0)) + 1
        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "source": source,
            "page": page,
            "document_type": doc.metadata.get(
                "document_type", "unknown"
            ),
            "excerpt": doc.page_content[:500],
        })

    return rows


# --------------------------------------------------
# Answer generation
# --------------------------------------------------

def answer_with_docs(question, docs):
    """Generate a grounded answer and return source references."""

    if not docs:
        return (
            "I couldn't find sufficient information "
            "in the provided documents.",
            [],
        )

    context = format_context(docs)

    print(f"Retrieved chunks sent to Gemini: {len(docs)}")
    print(f"Context length: {len(context)} characters")

    answer = get_chain().invoke({
        "question": question,
        "context": context,
    })

    return answer, source_rows(docs)
