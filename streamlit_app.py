import streamlit as st

from v4_reranking import load_or_create_index, ask


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Healthcare Policy RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- CUSTOM CSS --------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 0% 0%, rgba(37,99,235,.10), transparent 32%),
            radial-gradient(ellipse at 100% 0%, rgba(13,148,136,.09), transparent 30%),
            #f4f7fb;
        color: #172b47;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102744 0%, #174a78 65%, #117c82 100%);
    }
    [data-testid="stSidebar"] * { color: #eef6ff !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #c6d8ec !important; }
    .block-container { max-width: 1400px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
    .main-header {
        position: relative; overflow: hidden;
        background: linear-gradient(118deg, #102744 0%, #174a78 58%, #0f8b8b 100%);
        padding: 38px 40px; border-radius: 24px; color: white;
        margin-bottom: 26px; box-shadow: 0 18px 42px rgba(19,55,91,.18);
    }
    .main-header:after {
        content: ""; position: absolute; width: 260px; height: 260px;
        border: 1px solid rgba(255,255,255,.18); border-radius: 50%;
        right: -45px; top: -125px;
        box-shadow: 0 0 0 28px rgba(255,255,255,.04), 0 0 0 60px rgba(255,255,255,.025);
    }
    .main-header h1 {
        color: white !important; font-size: clamp(2rem, 4vw, 3.1rem);
        letter-spacing: -.04em; line-height: 1.1; margin: 0 0 12px 0;
    }
    .main-header p { color: #e0edf8 !important; font-size: 1rem; line-height: 1.7; margin: 0; max-width: 850px; }
    .info-card, .answer-card {
        background: #fff; color: #172b47; padding: 22px;
        border-radius: 18px; border: 1px solid #dce7f3;
        box-shadow: 0 9px 28px rgba(29,58,92,.05); margin-bottom: 14px;
    }
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border: 1px solid #dce7f3;
        border-radius: 15px; box-shadow: 0 6px 18px rgba(29,58,92,.04);
    }
    div[data-testid="stMetricLabel"] { color: #62738a; }
    div[data-testid="stMetricValue"] { color: #14233b; font-weight: 800; }
    .stButton button, .stFormSubmitButton button {
        border-radius: 11px; font-weight: 750; min-height: 2.8rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        transform: translateY(-1px); box-shadow: 0 8px 18px rgba(37,99,235,.14);
    }
    [data-testid="stExpander"] { background: rgba(255,255,255,.9); border: 1px solid #dce7f3; border-radius: 13px; }
    div[data-baseweb="select"] > div, .stTextArea textarea { border-radius: 12px !important; border-color: #d3dfed !important; }
    .stMarkdown, p, label { color: #263b55; }
    .disclaimer {
        color: #6a5a2a; background: #fff9e9; border: 1px solid #f0dfae;
        border-radius: 13px; padding: 13px 16px; font-size: .85rem; line-height: 1.55;
    }
    .footer { color: #78879a; text-align: center; padding: 18px 4px; font-size: .8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD FAISS INDEX ONCE --------------------

@st.cache_resource(show_spinner="Loading V4 FAISS index and reranker...")
def get_database():
    return load_or_create_index()


# -------------------- EXAMPLE QUESTIONS --------------------

EXAMPLE_QUESTIONS = [
    "What is the purpose of the Global Health Care policy?",
    "Which treatments require prior approval?",
    "What is the process for cashless hospitalization?",
    "What are the policy exclusions for pre-existing diseases?",
    "What are the conditions for emergency treatment outside my area of cover?",
    "What is the difference between inpatient and day-care treatment?",
    "What expenses are covered under living donor medical costs?",
]


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="main-header">
        <div style="color:#b8f3ed;font-size:.76rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:12px">AI · DOCUMENT INTELLIGENCE · V4</div><h1>Understand your health policy.<br>One question at a time.</h1>
        <p>
            AI-POWERED POLICY EXPLORATION &nbsp;|&nbsp;
            FAISS Retrieval + Cross-Encoder Reranking
            &nbsp;|&nbsp; Evidence-led answers with source references
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("📚 Policy Assistant")
    st.caption("Advanced Retrieval-Augmented Generation")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Ask a Question", "How It Works", "About"],
    )

    st.divider()

    st.subheader("Current Version")
    st.success("V4 — Cross-Encoder Reranking")

    st.markdown(
        """
        **Technology stack**
        - LangChain
        - FAISS vector search
        - Sentence Transformers
        - Cross-Encoder reranker
        - Gemini LLM
        - Streamlit
        """
    )

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.rerun()


# -------------------- HOW IT WORKS PAGE --------------------

if page == "How It Works":
    st.title("How the RAG pipeline works")

    st.markdown(
        """
        ### Retrieval pipeline

        1. **Question input:** The user submits a healthcare policy question.
        2. **Embedding:** The question is represented as a vector.
        3. **FAISS retrieval:** Similar document chunks are retrieved.
        4. **Cross-Encoder reranking:** Candidate chunks are scored for
           relevance to the question.
        5. **Answer generation:** The selected chunks are passed to the
           existing LLM chain in `rag_common.py`.
        6. **Source display:** The application displays the returned source
           documents and PDF page references.

        ### Important limitation

        Reranking improves the ordering of retrieved chunks, but it cannot
        guarantee correctness. Always verify important coverage decisions
        against the original policy wording and schedule.
        """
    )

    st.stop()


# -------------------- ABOUT PAGE --------------------

if page == "About":
    st.title("About this project")

    st.markdown(
        """
        The Healthcare Policy RAG Assistant answers questions using the
        healthcare policy documents configured in the project's `data`
        directory.

        **Core components**
        - Document loading and chunking
        - Semantic retrieval using FAISS
        - Cross-Encoder relevance scoring
        - LLM-based grounded responses
        - Source references for verification

        The assistant is an informational tool. It does not guarantee
        claim approval or replace confirmation from the insurer.
        """
    )

    st.stop()


# -------------------- MAIN QUESTION PAGE --------------------

st.subheader("Ask a Policy Question")

st.write(
    "Ask about coverage, exclusions, prior approval, hospitalization, "
    "claims, or policy limits."
)

selected_example = st.selectbox(
    "Choose an example question (optional)",
    ["Select an example..."] + EXAMPLE_QUESTIONS,
)

default_question = (
    selected_example
    if selected_example != "Select an example..."
    else ""
)

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        value=default_question,
        placeholder=(
            "Example: What conditions apply to emergency medical "
            "treatment outside my area of cover?"
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "🔎 Ask the Assistant",
        type="primary",
        use_container_width=True,
    )


# -------------------- INITIALIZE HISTORY --------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------- PROCESS QUESTION --------------------

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            db = get_database()

            with st.spinner(
                "Searching policy documents, reranking results, "
                "and generating your answer..."
            ):
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

            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "answer_time": answer_time,
                "total_time": total_time,
            }

            st.session_state.chat_history.append(result)

        except Exception as exc:
            st.error(f"Unable to process the question: {exc}")


# -------------------- DISPLAY LATEST ANSWER --------------------

if st.session_state.chat_history:
    result = st.session_state.chat_history[-1]

    st.divider()
    st.subheader("Answer")

    st.markdown(f"**Your question:** {result['question']}")

    with st.container(border=True):
        st.markdown(result["answer"])

    # Performance metrics
    st.subheader("Pipeline Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Retrieved Chunks",
        result["candidate_count"],
    )

    col2.metric(
        "Reranked Chunks",
        result["selected_count"],
    )

    col3.metric(
        "Retrieval + Reranking",
        f"{result['retrieval_time'] + result['rerank_time']:.2f}s",
    )

    col4.metric(
        "Total Time",
        f"{result['total_time']:.2f}s",
    )

    with st.expander("View detailed timing"):
        st.write(
            f"FAISS retrieval: {result['retrieval_time']:.2f} seconds"
        )
        st.write(
            f"Cross-Encoder reranking: {result['rerank_time']:.2f} seconds"
        )
        st.write(
            f"LLM answer generation: {result['answer_time']:.2f} seconds"
        )

    # Source references
    st.subheader("📄 Source Documents")

    if result["sources"]:
        for i, source in enumerate(result["sources"], start=1):
            source_name = source.get("source", "Unknown document")
            source_page = source.get("page", "Unknown")
            document_type = source.get("document_type", "Unknown")

            with st.expander(
                f"{i}. {source_name} — PDF page {source_page}"
            ):
                st.write(f"**Document type:** {document_type}")

                excerpt = source.get("excerpt")
                if excerpt:
                    st.write("**Retrieved excerpt:**")
                    st.write(excerpt)

                st.caption(
                    "Verify this passage against the original PDF "
                    "before making a coverage decision."
                )
    else:
        st.info("No source references were returned for this answer.")

    # Previous questions
    if len(st.session_state.chat_history) > 1:
        with st.expander("Previous questions"):
            for previous in reversed(
                st.session_state.chat_history[:-1]
            ):
                st.markdown(f"**Q:** {previous['question']}")
                st.markdown(f"**A:** {previous['answer']}")
                st.divider()

else:
    st.info(
        "Enter a question above to retrieve relevant policy clauses "
        "and generate an answer with source references."
    )


# -------------------- FOOTER --------------------

st.divider()

st.markdown(
    '<div class="disclaimer"><strong>Important:</strong> Educational information only. This assistant is not an official insurer service and cannot confirm coverage or claim approval. Verify important details against the applicable policy wording and schedule.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer">Healthcare Policy RAG Assistant · LangChain · FAISS · Cross-Encoder · Streamlit</div>',
    unsafe_allow_html=True,
)