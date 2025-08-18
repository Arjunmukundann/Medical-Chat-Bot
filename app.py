from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()
pinecone_api_key = os.environ.get('pinecone_api_key')
groq_api_key = os.environ.get('groq_api_key')

os.environ["pinecone_api_key"] = pinecone_api_key
os.environ["GROQ_API_KEY"] = groq_api_key   # ✅ correct key for langchain_groq

# Load embeddings + Pinecone index
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chat model
chatModel = ChatGroq(model="llama-3.3-70b-versatile")  # ✅ correct model name
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])  # Updated endpoint
def chat():
    try:
        data = request.get_json()
        msg = data.get("message", "")
        print("User  input:", msg)

        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])

        return jsonify({"response": response["answer"]})  # Ensure the key matches
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear", methods=["POST"])  # Added clear endpoint
def clear():
    # Implement your clear logic here if needed
    return jsonify({"status": "success", "message": "Chat cleared."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

    # Ensure the Pinecone index is created if it doesn't exist
