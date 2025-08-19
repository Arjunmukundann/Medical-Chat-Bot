from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from src.prompt import *
import os
from flask_cors import CORS

# Load environment variables first
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Get API keys from environment with validation
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
groq_api_key = os.environ.get('GROQ_API_KEY')

# Validate environment variables
if not pinecone_api_key:
    print("⚠️ PINECONE_API_KEY not found")
if not groq_api_key:
    print("⚠️ GROQ_API_KEY not found")

# Set environment variables (only if they exist)
if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Global variable for lazy initialization
rag_chain = None

def initialize_rag_chain():
    """Lazy initialization of RAG chain to reduce cold start time"""
    global rag_chain
    
    if rag_chain is not None:
        return rag_chain
    
    if not (pinecone_api_key and groq_api_key):
        print("❌ Missing API keys for initialization")
        return None
    
    try:
        print("🔄 Initializing RAG chain...")
        
        # Import only when needed to reduce startup time
        from src.helper import download_hugging_face_embeddings
        from langchain_pinecone import PineconeVectorStore
        from langchain_groq import ChatGroq
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        
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
        
        print("✅ RAG chain initialized successfully")
        return rag_chain
        
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {str(e)}")
        return None

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "rag_chain_ready": rag_chain is not None,
        "environment_ready": bool(pinecone_api_key and groq_api_key)
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Initialize RAG chain on first use (lazy loading)
        chain = initialize_rag_chain()
        
        if chain is None:
            return jsonify({
                "error": "Service is initializing or missing configuration. Please try again in a moment."
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        msg = data.get("message", "")
        
        if not msg.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        print(f"📥 User input: {msg}")
        
        response = chain.invoke({"input": msg})
        print(f"📤 Response: {response['answer']}")
        
        return jsonify({"response": response["answer"]})
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route("/api/clear", methods=["POST"])
def clear():
    return jsonify({"status": "success", "message": "Chat cleared."})

# Vercel serverless function entry point
def handler(event, context):
    """AWS Lambda/Vercel handler"""
    return app

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)