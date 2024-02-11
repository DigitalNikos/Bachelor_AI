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

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)


