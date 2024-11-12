from langchain_ollama import OllamaLLM

llm = OllamaLLM(model='llama3.1')
messages = [{'role': 'user', 'content': 'Name an engineer that passes the vibe check'}]
stream = llm.chat(model='llama3.1', messages=messages, stream=True)

llm

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)