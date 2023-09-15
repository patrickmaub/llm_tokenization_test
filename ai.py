import openai
import numpy as np
import tiktoken
import os
enc = tiktoken.encoding_for_model("gpt-4")
openai.api_key = os.environ["OPENAI_API_KEY"]

def completion(messages=[], max_tokens=1000, temperature=0.7, model="gpt-3.5-turbo", stream=False):
    response = openai.ChatCompletion.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
    if stream:
        return response
    response_str = ""
    for i in range(len(response['choices'])):
        response_str += response['choices'][i]['message']['content']
    return response_str
