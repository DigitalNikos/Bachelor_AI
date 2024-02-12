# Vector store
import chromadb
# from chromadb.config import Settings
# from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings

# persist_directory = 'db'

# # HuggingFace embedding is free!
# embedding = HuggingFaceEmbeddings()

# vectordb = Chroma(persist_directory=persist_directory, 
#                   embedding_function=embedding)


# retriever = vectordb.as_retriever(search_kwargs={"k": 5})

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="Documents/external data source")