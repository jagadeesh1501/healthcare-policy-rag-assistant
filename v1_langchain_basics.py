import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY is missing. Add your Gemini API key to the .env file.")
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. This V1 demo has no document retrieval. For exact policy coverage, say the policy document must be checked."),
    ("human", "{question}")
])
parser = StrOutputParser()
# LCEL Runnable chain: Prompt -> LLM -> Output Parser
chain = prompt | llm | parser

if __name__ == "__main__":
    q = input("Ask a question (or press Enter for a RAG definition): ").strip() or "What is RAG, in simple terms?"
    print("\nANSWER:\n", chain.invoke({"question": q}))
