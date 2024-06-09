from .vectordb import get_vectorstore
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = get_vectorstore()
def rag_request(query: str, model: str) -> dict:
    retriever = vector_store.similarity_search(query)
    context = " ".join(retriever)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Contexto: {context}\nPregunta: {query}\nRespuesta:"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()