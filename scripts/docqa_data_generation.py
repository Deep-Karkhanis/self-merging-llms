import openai
from tqdm import tqdm
# from retry import retry
import time
import random
import pickle

# openai.api_key = "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"
openai.api_key = "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"

DATA_SIZE = 10000
SYS_PROMPT = """You will be asked to generate documents, questions and answers. You should first generate between 1 to 5 documents which are related to each other. You should then generate questions which ask for information contained in the documents. The questions might be such that they ask for information from multiple documents and not just one. You should then generate answers to all these questions. The answers you generate should only use information from the generated documents. You should reference the document titles in your answer"""

USR_PROMPT_DOC_1 = "Pick a number between 3-5. Generate that many documents which are related to "
USR_PROMPT_DOC_2 = """. Write <STOP> when done
 
Use the following format for generation:
Documents:
1. <Document Title 1>
<Document Text 1>

2. <Document Title 2>
<Document Text 2>
...

<STOP>"""

USR_PROMPT_Q = """Generate 10 questions which ask for information contained in the documents. Questions should be such that it asks for information from multiple documents and not just one. Write <STOP> when done

Use the following format for generation:
Questions:
1. <Question 1>
(<Source Document Title 1>, <Source Document Title 2>, ...)
2. <Question 2> 
(<Source Document Title 1>, <Source Document Title 2>, ...)
...

<STOP>"""

USR_PROMPT_ANS="""Generate answers to all these questions. The answers should only use information from the generated documents. Write <STOP> when done

Use the following format for generation:
Answers:
1. <Answer 1>
(<Reference 1>, <Reference 2>, ...)
2. <Answer 2>
(<Reference 1>, <Reference 2>, ...)
...

<STOP>"""

FILE_NAME = "docQA_data_gpt4.txt"
MODEL = "gpt-3.5-turbo"
FIELD_FILE = "/home/ubuntu/llm-training/data/docQA_fields_gpt4.txt"
SAVE_DIR = "/home/ubuntu/llm-training/data/docQA_data/run2"

FIELDS = []
with open(FIELD_FILE) as f:
    lines = f.readlines()
    FIELDS = [l.strip().split('. ')[-1] for l in lines]

# @retry(tries=3, delay=5, backoff=2, exceptions=[openai.error.RateLimitError, openai.error.ServiceUnavailableError])
def openai_query(messages):
    retries = 0
    full_response = ""
    while True:
        response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages
            )
        str_response = response.choices[0].message.content
        lines = str_response.strip().split('\n')
        messages.append({"role": "assistant", "content": str_response})
        full_response += str_response
        
        if "<STOP>" not in lines[-1]:
            retries += 1
            print("<STOP> not received")
            # print(messages[-1]['content'])
            print("retring...")
            if retries == 5:
                raise Exception("Exceeded retry limit to get a <STOP>")
            continue
        else:
            break
    return messages, full_response

def openai_chat(FIELD_NAME):
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": USR_PROMPT_DOC_1+FIELD_NAME+USR_PROMPT_DOC_2}
    ]
    data_point = {}
    data_point["Field"] = FIELD_NAME

    messages, full_response = openai_query(messages)
    data_point["Docs"] = full_response
    # print("Docs generated:\n")
    # print(messages[-1]['content'])
    
    messages.append({"role": "user", "content": USR_PROMPT_Q})
    messages, full_response = openai_query(messages)
    data_point["Questions"] = full_response
    # print("Qs generated:\n")
    # print(messages[-1]['content'])

    messages.append({"role": "user", "content": USR_PROMPT_ANS})
    messages, full_response = openai_query(messages)
    data_point["Answers"] = full_response
    # print("Ans generated:\n")
    # print(messages[-1]['content'])

    return data_point

for idx in tqdm(range(DATA_SIZE)):
    try:
        data_point = openai_chat(random.choice(FIELDS))    
        with open(SAVE_DIR+'/'+str(idx)+'.pkl', 'wb') as f:
            pickle.dump(data_point, f)
    except:
        print("An exception occurred")
        time.sleep(40)
        continue