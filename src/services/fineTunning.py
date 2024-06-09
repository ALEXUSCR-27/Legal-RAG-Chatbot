import os
from openai import OpenAI

def fine_tunning():
    client = OpenAI()

    client.files.create(
    file=open("codigo_penal_2019_qa_extended_new_format.jsonl", "rb"),
    purpose="fine-tune"
    )

    client.fine_tuning.jobs.create(
    training_file="file-wOXUgDX1Mv2ARZPwEdm3E2yU", 
    model="gpt-3.5-turbo"
    )
    return "Model fine-tunned", None
