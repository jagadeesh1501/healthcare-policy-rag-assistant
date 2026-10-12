
import time
from langchain_community.vectorstores import FAISS

from rag_common import (
    get_embeddings,
    answer_with_docs,
    RETRIEVER_K,
    VECTOR_DIR,
)

# Keep V2's FAISS index separate from other versions
FAISS_DIR = VECTOR_DIR / "faiss_v2"


def load_or_create_index():
    """Load the saved FAISS index or create a new one."""
    embeddings = get_embeddings()

    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and metadata_file.exists():
        print("Loading existing FAISS index...")

        db = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return db

    print("First run: loading PDFs and creating FAISS index...")

    from rag_common import load_chunks

    start = time.perf_counter()
    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks in "
        f"{time.perf_counter() - start:.2f}s"
    )

    if not chunks:
        raise ValueError(
            "No document chunks found. Check your PDF_DIR and PDF files."
        )

    start = time.perf_counter()
    db = FAISS.from_documents(chunks, embeddings)

    print(
        f"Created embeddings and index in "
        f"{time.perf_counter() - start:.2f}s"
    )

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(FAISS_DIR))

    print(f"Index saved to: {FAISS_DIR.resolve()}")

    return db


def retrieve_relevant_docs(question, db):
    """
    Search FAISS using multiple question phrasings.
    Deduplicate matching chunks before sending them to Gemini.
    """

    queries = [
        question,
        "purpose and objective of the health insurance policy",
        "scope of cover, medical expenses and healthcare benefits",
        "what hospitalization and medical treatment expenses are covered",
    ]

    unique_docs = {}

    for query in queries:
        results = db.similarity_search(
            query,
            k=3,
        )

        for doc in results:
            key = (
                doc.metadata.get("source", "unknown"),
                doc.metadata.get("page", 0),
                doc.page_content,
            )

            unique_docs[key] = doc

    docs = list(unique_docs.values())

    print(f"\nUnique chunks retrieved: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        page = int(doc.metadata.get("page", 0)) + 1

        print(f"\nCHUNK {i}")
        print("Source:", doc.metadata.get("source", "unknown"))
        print("PDF page:", page)
        print(
            "Document type:",
            doc.metadata.get("document_type", "unknown"),
        )
        print("Content:")
        print(doc.page_content[:700])
        print("-" * 60)

    return docs


def ask(question, db):
    """Retrieve relevant chunks and generate a grounded answer."""

    # Step 1: Retrieve using multiple queries
    start = time.perf_counter()

    docs = retrieve_relevant_docs(question, db)

    retrieval_time = time.perf_counter() - start

    if not docs:
        return (
            "I couldn't find sufficient information in the provided documents.",
            [],
            [],
            retrieval_time,
            0.0,
        )

    # Step 2: Generate the answer using the existing RAG chain
    start = time.perf_counter()

    try:
        answer, sources = answer_with_docs(question, docs)
    except Exception as exc:
        generation_time = time.perf_counter() - start

        print(f"\nGemini/chain error: {exc}")
        print(f"Failed after: {generation_time:.2f}s")

        raise

    generation_time = time.perf_counter() - start

    return (
        answer,
        sources,
        docs,
        retrieval_time,
        generation_time,
    )


if __name__ == "__main__":
    try:
        print("FAISS index folder:", FAISS_DIR.resolve())
        print("Retrieval count per query: 3")
        print("Multi-query retrieval: enabled")

        db = load_or_create_index()

        while True:
            question = input(
                "\nAsk a policy question (or type exit): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                print("Exiting Healthcare Policy RAG Assistant.")
                break

            if not question:
                continue

            try:
                (
                    answer,
                    sources,
                    docs,
                    retrieval_time,
                    generation_time,
                ) = ask(question, db)

                print(f"\nANSWER:\n{answer}")

                print(
                    f"\nRetrieval time: {retrieval_time:.2f}s"
                )
                print(
                    f"Gemini/chain time: {generation_time:.2f}s"
                )

                print("\nSOURCES:")

                if not sources:
                    print("No sources returned.")
                else:
                    for source in sources:
                        print(
                            f"- {source['source']} | "
                            f"PDF page {source['page']} | "
                            f"{source['document_type']}"
                        )

            except Exception as exc:
                print(f"\nCould not answer this question: {exc}")

    except Exception as exc:
        print(f"\nApplication startup error: {exc}")
        raise
