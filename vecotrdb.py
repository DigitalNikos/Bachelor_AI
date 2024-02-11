# Vector store
import chromadb
from chromadb.config import Settings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

persist_directory = 'db'

# HuggingFace embedding is free!
embedding = HuggingFaceEmbeddings()

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="db/"))

# client.delete_collection(name="Pdf")
try:
    client.delete_collection(name="Documents")
except IndexError as e:
    print(f"Error deleting collection: {e}")

collection = client.create_collection(
    name="Documents",
    embedding_function=embedding,
    metadata={"hnsw:space": "cosine"}
    )

langchain_chroma = Chroma(
    client=client,
    collection_name="Documents",
    embedding_function=embedding,
)
