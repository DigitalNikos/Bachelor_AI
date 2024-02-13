import streamlit as st
import os
import logging

from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from validation import is_valid_url, validate_file
from loadfiles import file_uploader
# from vecotrdb import retriever
from vecotrdb import collection
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp


logging.basicConfig(filename='app.log', level=logging.INFO)

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

llama_model_path = 'llama-2-7b-chat.Q5_0.gguf'

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# for token-wise streaming so you'll see the answer gets generated token by token when Llama is answering your question
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')    

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    url = st.sidebar.text_input("Enter a URL")
    st_document = st.sidebar.file_uploader("Upload a File")
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')


    llm = LlamaCpp(
      model_path=llama_model_path,
      temperature=temperature,
      top_p=top_p,
      n_ctx=4096,
      n_gpu_layers=n_gpu_layers,
      n_batch=n_batch,
      callback_manager=callback_manager,
      verbose=True,
    )

    if validate_file(st, st_document):
        
        # Process the file if validation is successful
        logging.info(f"File is valid")
        file_uploader(st,st_document)
    # if st_document is not None:
    #    

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    question = prompt_input
    # qa_chain = RetrievalQA.from_chain_type(
    # llm,
    # retriever=retriever
    # )
    # llm_response = qa_chain(prompt_input)
    results = collection.query(query_texts=[prompt_input], n_results=1)
    retrieved_documents = results['documents'][0]
    for document in retrieved_documents:
      logging.info(f"document:{document}")
    concatenated_documents = ' '.join(retrieved_documents)
    concatenated_documents = concatenated_documents.replace("{", "").replace("}","")
    # logging.info(f"Answer from DB:{llm_response}")
    pre_prompt = f"""<s>[INST] <<SYS>>Use the following context information to answer the question at the end. If you don't know the answer, just say you don't know. 
            Do not try to invent an answer.
  
            """
    # pre_prompt = """[INST] <<SYS>>
    #               You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.

    #               If you cannot answer the question from the given documents, please state that you do not have an answer.\n
    #               """
    
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=retriever
    # )

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            pre_prompt += "User: " + dict_message["content"] + "\n\n"
            dictmeddage= dict_message["content"] 
            logging.info(f"App.log: dict_message[content]: {dictmeddage}")
        else:
            pre_prompt += "Assistant: " + dict_message["content"] + "\n\n"

    prompt = pre_prompt +  f""" Context:{concatenated_documents} Question: {prompt_input} <</SYS>>User : {question}""" + "[\INST]"
    logging.info(f"App.log: prompt before: {prompt}")
    llama_prompt = PromptTemplate(template=prompt, input_variables=["question", "prompt_input", "concatenated_documents"])

    

    chain = LLMChain(llm=llm, prompt=llama_prompt)

    result = chain({
                "question": question,
                "prompt_input": prompt_input,
                "concatenated_documents": concatenated_documents
                 })


    return result['text']

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)