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
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Get API keys from environment
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
groq_api_key = os.environ.get('GROQ_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize components (wrap in try-catch for better error handling)
try:
    embeddings = download_hugging_face_embeddings()
    index_name = "medicalbot"
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chatModel = ChatGroq(model="llama-3.3-70b-versatile")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("✅ All components initialized successfully")
    
except Exception as e:
    print(f"❌ Error initializing components: {str(e)}")
    rag_chain = None

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "rag_chain_ready": rag_chain is not None})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        if rag_chain is None:
            return jsonify({"error": "Chatbot not properly initialized"}), 500
            
        data = request.get_json()
        msg = data.get("message", "")
        
        if not msg.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        print("User input:", msg)
        
        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])
        
        return jsonify({"response": response["answer"]})
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route("/api/clear", methods=["POST"])
def clear():
    return jsonify({"status": "success", "message": "Chat cleared."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)