from dotenv import load_dotenv
import os
from src.helpers import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load documents
extracted_data = load_pdf_files(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_splitter(filter_data)

# Embeddings
embedding = download_embeddings()

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "floatchat-research"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Store vectors
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name,
)
print("✅ Documents indexed in Pinecone successfully!")
