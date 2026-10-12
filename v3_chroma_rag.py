
import time

from langchain_chroma import Chroma

from rag_common import (
    load_chunks,
    get_embeddings,
    answer_with_docs,
    VECTOR_DIR,
    RETRIEVER_K,
)

# Keep V3 separate from V2's FAISS index
CHROMA_DIR = VECTOR_DIR / "chroma_v3"
COLLECTION_NAME = "healthcare_policy"


def load_or_create_index():
    """Reuse the persistent ChromaDB collection or create it once."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    # Open the existing persistent collection
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Reuse the collection if it already contains documents
    existing_count = db._collection.count()

    if existing_count > 0:
        print(f"Loaded existing ChromaDB index: {CHROMA_DIR.resolve()}")
        print(f"Existing chunks: {existing_count}")
        return db

    # Build the index only if the collection is empty
    print("Creating ChromaDB index for the first time...")

    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            "No document chunks found. Check your PDF_DIR and PDF files."
        )

    print(f"Loaded {len(chunks)} chunks.")

    # Add chunks in batches to avoid oversized requests
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        db.add_documents(batch)
        print(
            f"Indexed {min(start + batch_size, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    print("ChromaDB index created successfully.")
    print(f"Index location: {CHROMA_DIR.resolve()}")

    return db


def retrieve_relevant_docs(question, db):
    """Retrieve the most relevant chunks from ChromaDB."""

    start = time.perf_counter()

    docs = db.similarity_search(
        question,
        k=RETRIEVER_K,
    )

    retrieval_time = time.perf_counter() - start

    print(f"\nRetrieved {len(docs)} chunks in {retrieval_time:.2f}s")

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
        print(doc.page_content[:500])
        print("-" * 60)

    return docs, retrieval_time


def ask(question, db):
    """Retrieve policy information and generate an answer."""

    docs, retrieval_time = retrieve_relevant_docs(question, db)

    if not docs:
        return (
            "I couldn't find sufficient information in the provided documents.",
            [],
            [],
            retrieval_time,
            0.0,
        )

    start = time.perf_counter()

    answer, sources = answer_with_docs(question, docs)

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
        print("ChromaDB index folder:", CHROMA_DIR.resolve())
        print("Collection:", COLLECTION_NAME)
        print("Retrieval count:", RETRIEVER_K)

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

                print(f"\nRetrieval time: {retrieval_time:.2f}s")
                print(f"Gemini generation time: {generation_time:.2f}s")

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
