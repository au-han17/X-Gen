import os
import openai
from bs4 import BeautifulSoup
import tiktoken
from tqdm import tqdm
import json
import time


openai.api_key = "" # add your own key

augmented = []


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chatgpt_completion(model_new="gpt",sys_msg='helper',prompt_new="Hello_World", temperature_new=0.05, top_p_new=1, n_new=1, max_tokens_new=100):
    Chat_Completion = openai.ChatCompletion.create(
        model=model_new,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt_new}
        ],
        temperature=temperature_new,
        top_p=top_p_new,
        n=n_new,
        max_tokens=max_tokens_new,
        presence_penalty=0,
        frequency_penalty=0
    )
    return Chat_Completion



list_of_text_contents = []
list_of_files = []

with open('annotation.json','r') as file:
    myjs = json.load(file)

train = myjs['train']
start_index = 0
pbar = tqdm(total=len(train))
step =1
while start_index < len(train):

    try:
        report = train[start_index]['report']
        idx = train[start_index]['id']
        prompt = report

        #num_tokens = num_tokens_from_string(prompt, "gpt2")
        #print(num_tokens)
        guide = "The follwoing is a chest X-Ray report. Rewrite the report in different styles while keep the medical terminologies intact.Three different styles are as follows:1. professional and concise, 2. formal and concise 3. formal and detailed. Make sure each style accounts for less than 120 words.\n"
        
        completion = chatgpt_completion(model_new="gpt-4",prompt_new=prompt,sys_msg= guide,max_tokens_new = 360,temperature_new= 0.1)
        rewrite_finding = completion.choices[0].message.content
        entry = {
            'ID':idx,
            'report': rewrite_finding
        }
        augmented.append(entry)
        file_name  = f"{start_index}" + "_augmented.txt"
        with open(file_name, "w") as f:
            f.write(rewrite_finding)
        pbar.update(step)
        start_index += step
    except Exception as e:
        print(f"Error encountered as {e}")
        print("Wait for 30s before retrying.")
        time.sleep(30)

pbar.close()
with open('iu_xray_aug.json', "w") as f:
    json.dump(augmented,f)
print("Finished.")
   






