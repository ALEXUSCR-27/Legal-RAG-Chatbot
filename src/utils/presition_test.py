import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#nltk.download('punkt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def word_overlap_evaluation(reference_text, generated_text):
    reference_text_list = reference_text.lower().split()
    generated_text_list = generated_text.lower().split()
    bleu_score = sentence_bleu([reference_text_list], generated_text_list)
    return bleu_score

def embedding_based_evaluation(reference_text, generated_text):
    reference_text = reference_text.lower()
    generated_text = generated_text.lower()
    if (len(reference_text)>1400): reference_text = reference_text[:1400]
    if (len(generated_text)>1400): generated_text = generated_text[:1400]
    reference_inputs = tokenizer(reference_text, return_tensors='pt')
    generated_inputs = tokenizer(generated_text, return_tensors='pt')

    with torch.no_grad():
        reference_outputs = model(**reference_inputs)
        generated_outputs = model(**generated_inputs)

    reference_embedding = reference_outputs.last_hidden_state[:, 0, :].numpy()
    generated_embedding = generated_outputs.last_hidden_state[:, 0, :].numpy()

    cosine_sim = cosine_similarity(reference_embedding, generated_embedding)[0][0]
    
    return cosine_sim


def clean_text(text):
    text = text.replace("{","").replace("}","")
    text = text.replace('"', "")
    return text

