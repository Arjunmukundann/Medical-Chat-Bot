from src.helper import load_pdf_file,split_data,download_embedding
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()

pinecone_api_key=os.environ.get("pinecone_api_key")
os.environ["pinecone_api_key"]=pinecone_api_key

extractor=load_pdf_file(data='C:/Users/Arjun/OneDrive/Desktop/DO IT/Medical-Chat-Bot/Data')

text_chunks=split_data(extractor)

embedding= download_embedding()


pc = Pinecone(api_key=pinecone_api_key)
index_name = "medibot"

# Only create the index if it does not already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")


vector=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embedding,
)