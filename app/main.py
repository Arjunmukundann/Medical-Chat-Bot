from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import os
import time
import logging

# Your existing imports adapted (you may need to adjust relative imports)
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# ---- Setup ----
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medibot")

load_dotenv()
pinecone_api_key = os.environ.get("pinecone_api_key")
groq_api_key = os.environ.get("groq_api_key")

if not pinecone_api_key or not groq_api_key:
    logger.warning("Missing API keys in environment. Check environment variables.")

os.environ["pinecone_api_key"] = pinecone_api_key or ""
os.environ["groq_api_key"] = groq_api_key or ""

# ---- Embedding / Vector Store Initialization ----
try:
    embeddings = download_embedding()
    index_name = "medibot"
    vector = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
except Exception:
    logger.exception("Failed to initialize vector store or embeddings.")
    vector = None
    retriever = None

# ---- LLM & Chain ----
chat_llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

from src.prompt import system_prompt  # ensure this path is correct

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

chain = create_stuff_documents_chain(chat_llm, prompt)
if retriever:
    rag_chain = create_retrieval_chain(retriever, chain)
else:
    rag_chain = None

# ---- Schemas ----
class ChatRequest(BaseModel):
    message: str

# ---- Routes ----
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.post("/api/clear")
async def clear():
    # Placeholder for clearing session/history if you implement it
    return {"status": "cleared"}

@app.post("/api/chat")
async def chat(payload: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Retrieval chain not initialized")

    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="empty message")

    logger.info(f"[DEBUG] User input: {user_message}")

    # 1. Retrieve documents
    try:
        retrieved_docs = retriever.get_relevant_documents(user_message)
        logger.info(f"[DEBUG] Retrieved {len(retrieved_docs)} docs. Sample titles/snippets: "
                    + "; ".join(
                          (doc.metadata.get("title", "no-title") if hasattr(doc, "metadata") else "no-meta")
                          for doc in retrieved_docs[:3]
                      ))
    except Exception as e:
        logger.exception("[ERROR] Retrieval failed")
        raise HTTPException(status_code=500, detail=f"retrieval error: {str(e)}")

    # 2. Invoke the RAG chain
    try:
        response_obj = rag_chain.invoke({"input": user_message})
        logger.info(f"[DEBUG] Raw chain output: {response_obj}")
    except Exception as e:
        logger.exception("[ERROR] Chain invocation failed")
        raise HTTPException(status_code=500, detail=f"chain invocation error: {str(e)}")

    # 3. Extract answer
    answer = ""
    if isinstance(response_obj, dict):
        answer = response_obj.get("answer") or response_obj.get("response") or response_obj.get("output") or ""
    else:
        answer = str(response_obj)

    if not answer:
        logger.warning("[WARN] Empty answer extracted, falling back to echo")
        answer = f"(No answer generated) You asked: {user_message}"

    logger.info(f"[DEBUG] Final answer: {answer}")
    return {"response": answer}
