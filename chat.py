from src.services.vectordb import get_vectorstore
from src.services.rag import rag_request
import gradio as gr
from src.utils.presition_test import embedding_based_evaluation, word_overlap_evaluation

def ask_models(query, history):
    
    response_openai_base = rag_request(query, "gpt-3.5-turbo")
    response_openai_fine_tuned = rag_request(query, "ft:gpt-3.5-turbo-0125:angel::9XylmSeE")
    semantic = embedding_based_evaluation(response_openai_base, response_openai_fine_tuned)
    print(f"Semantic Presition: {semantic}")
    final_response = f"Response without fine-tunning:\n{response_openai_base}\n\nResponse with fine-tunning:\n{response_openai_fine_tuned}\n"

    return final_response

gr.ChatInterface(
    ask_models,
    chatbot=gr.components.Chatbot(label=None, height=650, width=700, elem_classes="chatbox"),
    title="Legal Assistant",
    css="src/css/chat.css",
    undo_btn=None,

).launch(share=True)
