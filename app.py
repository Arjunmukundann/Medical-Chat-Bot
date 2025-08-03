from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import logging
import time
import threading

# ---- Setup ----
app = Flask(__name__, template_folder="templates")  # ensure index.html is in templates/
CORS(app)  # for dev; in production restrict origins

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medibot")

load_dotenv()
pinecone_api_key = os.environ.get("pinecone_api_key")
groq_api_key = os.environ.get("groq_api_key")

if not pinecone_api_key or not groq_api_key:
    logger.warning("Missing API keys in environment. Check .env file.")

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

# Import your prompt definitions
from src.prompt import system_prompt  # adjust if needed

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

# ---- Routes ----

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), 200


@app.route("/api/clear", methods=["POST"])
def clear():
    # If you maintain session/history, clear it here; placeholder:
    return jsonify({"status": "cleared"}), 200
@app.route("/api/chat", methods=["POST"])
def chat():
    if not rag_chain:
        return jsonify({"error": "retrieval chain not initialized"}), 500

    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "empty message"}), 400

        logger.info(f"[DEBUG] User input: {user_message}")

        # 1. Retrieve documents manually to see what's coming back
        try:
            retrieved_docs = retriever.get_relevant_documents(user_message)
            logger.info(f"[DEBUG] Retrieved {len(retrieved_docs)} docs. Sample titles/snippets: "
                        + "; ".join(
                              (doc.metadata.get("title", "no-title") if hasattr(doc, "metadata") else "no-meta")
                              for doc in retrieved_docs[:3]
                          ))
        except Exception as e:
            logger.exception("[ERROR] Retrieval failed")
            return jsonify({"error": f"retrieval error: {str(e)}"}), 500

        # 2. Invoke the full RAG chain
        try:
            response_obj = rag_chain.invoke({"input": user_message})
            logger.info(f"[DEBUG] Raw chain output: {response_obj}")
        except Exception as e:
            logger.exception("[ERROR] Chain invocation failed")
            return jsonify({"error": f"chain invocation error: {str(e)}"}), 500

        # 3. Extract answer safely
        answer = None
        if isinstance(response_obj, dict):
            answer = response_obj.get("answer") or response_obj.get("response") or response_obj.get("output") or ""
        else:
            answer = str(response_obj)

        if not answer:
            logger.warning("[WARN] Empty answer extracted, falling back to echo")
            answer = f"(No answer generated) You asked: {user_message}"

        logger.info(f"[DEBUG] Final answer: {answer}")
        return jsonify({"response": answer}), 200

    except Exception as e:
        logger.exception("Unhandled exception in /api/chat")
        return jsonify({"error": "internal server error"}), 500



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
