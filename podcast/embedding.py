
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.docstore.document import Document

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pinecone_index_name = "podcast-transcripts"

def store_embedding(documents):
    docsearch = PineconeVectorStore.from_documents(documents , embedding_function , index_name = pinecone_index_name)
    return docsearch