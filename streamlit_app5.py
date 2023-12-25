#Import necessary modules
import streamlit as st
from langchain.chains import RetrievalQA
import textwrap

# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Function to process QA chain response
def process_qa_chain_response(qa_chain_response):
    result = qa_chain_response.get('result', 'No result')
    sources = qa_chain_response.get('source_documents', [])

    print(wrap_text_preserve_newlines(result))
    print('\n\nSources:')
    for source in sources:
        print(source.get('metadata', {}).get('source', 'Unknown source'))

# Function to load your QA chain
def load_qa_chain():
    config = {'max_new_tokens': 1024, 'temperature': 0, 'context_length': 1024}
    llm = CTransformers(model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config, n_ctx=2048)
    # Load embeddings from file
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    # Create Chroma instance with loaded embeddings
    vectordb = Chroma(embedding_matrix=embeddings.embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain

# Load your QA chain
qa_chain = load_qa_chain()

# StreamLit UI
st.set_page_config(page_title="Intelligent Customer Chatbot", page_icon=":robot:")
st.header("Intelligent Customer Chatbot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Function to get user input
def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

# Get user input
user_input = get_text()

if user_input:
    # Run your QA chain with user input
    qa_chain_response = qa_chain.run(input=user_input)

    # Store user input and QA chain response in session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(qa_chain_response)

# Display past interactions
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        process_qa_chain_response(st.session_state["generated"][i])
        print('\nUser Input:')
        print(st.session_state["past"][i])
        print('\n' + '-'*50 + '\n')
