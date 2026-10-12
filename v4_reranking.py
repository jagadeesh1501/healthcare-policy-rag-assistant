
import time
from pathlib import Path

from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS

from rag_common import (
    load_chunks,
    get_embeddings,
    answer_with_docs,
    VECTOR_DIR,
    RETRIEVER_K,
    RERANK_TOP_N,
)


# -------------------- CONFIGURATION --------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Keep V4's index separate from the V2 FAISS index.
FAISS_DIR = Path(VECTOR_DIR) / "faiss_v4"

# Retrieve more candidates before reranking.
CANDIDATE_K = max(RETRIEVER_K, 15)


# -------------------- LOAD RERANKER ONCE --------------------

print(f"Loading reranker model: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)


# -------------------- FAISS INDEX --------------------

def load_or_create_index():
    """
    Load the saved V4 FAISS index if available.
    Otherwise, create the index from the policy PDF chunks and save it.
    """

    embeddings = get_embeddings()

    if (
        FAISS_DIR.exists()
        and (FAISS_DIR / "index.faiss").exists()
        and (FAISS_DIR / "index.pkl").exists()
    ):
        print(f"\nLoading existing FAISS index: {FAISS_DIR}")

        # Only load an index created and trusted by you.
        db = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        print("Existing V4 FAISS index loaded successfully.")
        return db

    print("\nCreating a new V4 FAISS index...")

    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            "No document chunks found. Check your PDF_DIR and policy PDFs."
        )

    print(f"Loaded {len(chunks)} document chunks.")

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(FAISS_DIR))

    print(f"V4 FAISS index saved to: {FAISS_DIR}")

    return db


# -------------------- RETRIEVAL --------------------

def retrieve_candidates(question, db):
    """Retrieve candidate chunks using FAISS similarity search."""

    total_chunks = db.index.ntotal
    k = min(CANDIDATE_K, total_chunks)

    if k == 0:
        return []

    return db.similarity_search(question, k=k)


# -------------------- CROSS-ENCODER RERANKING --------------------

def rerank(question, docs, top_n=RERANK_TOP_N):
    """
    Score each question-document pair with the Cross-Encoder.
    Return the highest-scoring documents first.
    """

    if not docs:
        return []

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    top_n = max(1, min(top_n, len(ranked_docs)))

    return [
        doc
        for score, doc in ranked_docs[:top_n]
    ]


# -------------------- ASK A QUESTION --------------------

def ask(question, db):
    """Retrieve, rerank, and generate an answer."""

    question = question.strip()

    if not question:
        raise ValueError("Please enter a policy question.")

    # Step 1: Retrieve candidate chunks.
    retrieval_start = time.perf_counter()
    candidates = retrieve_candidates(question, db)
    retrieval_time = time.perf_counter() - retrieval_start

    if not candidates:
        return (
            "I couldn't find sufficient information in the provided documents.",
            [],
            [],
            [],
            retrieval_time,
            0.0,
            0.0,
        )

    # Step 2: Rerank the retrieved chunks.
    rerank_start = time.perf_counter()
    selected = rerank(question, candidates)
    rerank_time = time.perf_counter() - rerank_start

    # Step 3: Generate an answer using the selected chunks.
    answer_start = time.perf_counter()
    answer, sources = answer_with_docs(question, selected)
    answer_time = time.perf_counter() - answer_start

    return (
        answer,
        sources,
        candidates,
        selected,
        retrieval_time,
        rerank_time,
        answer_time,
    )


# -------------------- MAIN PROGRAM --------------------

if __name__ == "__main__":
    try:
        db = load_or_create_index()

        print("\nHealthcare Policy RAG Assistant — V4")
        print("FAISS retrieval + Cross-Encoder reranking")
        print("Type 'exit' to stop.\n")

        while True:
            question = input("Ask a policy question: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Exiting Healthcare Policy RAG Assistant.")
                break

            if not question:
                print("Please enter a question.\n")
                continue

            try:
                (
                    answer,
                    sources,
                    candidates,
                    selected,
                    retrieval_time,
                    rerank_time,
                    answer_time,
                ) = ask(question, db)

                total_time = (
                    retrieval_time + rerank_time + answer_time
                )

                print("\n" + "=" * 60)
                print("ANSWER")
                print("=" * 60)
                print(answer)

                print("\n" + "-" * 60)
                print("RETRIEVAL AND RERANKING DETAILS")
                print("-" * 60)

                print(f"Candidate chunks retrieved : {len(candidates)}")
                print(f"Chunks selected by reranker: {len(selected)}")
                print(f"FAISS retrieval time       : {retrieval_time:.2f}s")
                print(f"Cross-Encoder rerank time  : {rerank_time:.2f}s")
                print(f"Answer generation time     : {answer_time:.2f}s")
                print(f"Total processing time      : {total_time:.2f}s")

                print("\n" + "-" * 60)
                print("SOURCES")
                print("-" * 60)

                if sources:
                    for source in sources:
                        print(
                            f"- {source['source']} | "
                            f"PDF page {source['page']} | "
                            f"{source['document_type']}"
                        )
                else:
                    print("No sources returned.")

                print()

            except Exception as exc:
                print(f"\nError processing question: {exc}\n")

    except Exception as exc:
        print(f"\nCould not start V4: {exc}")
        print("Check your dependencies, configuration, and PDF files.")
