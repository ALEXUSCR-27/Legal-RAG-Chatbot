from .services.vectordb import get_vectorstore
import os
vector_store = get_vectorstore()

def create_collection(collection_name):
    status = vector_store.create_collection(collection_name=collection_name)
    print(status)
    return None

def save_doc_vectors(collection_name, filepath):
    status = vector_store.create_records(collection_name=collection_name, filepath=filepath)
    print(status)
    return None
