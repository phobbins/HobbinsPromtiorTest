import streamlit as st
from streamlit_chat import message
from langchain_community.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import pickle


st.set_page_config(page_title='Llama2-Chatbot for Promtior')
st.header('Llama2-Powered Chatbot for Promtior Test:robot_face:')

def load_llm():

    # load the llm with ctransformers
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q2_K.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0})
    return llm


def load_vector_store():

    # load the vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    db = FAISS.load_local("faiss", embeddings)
    return db


def load_prompt_template():

    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

    return prompt 

def create_qa_chain():

    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store()
    prompt = load_prompt_template()

    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    
    return qa_chain

def generate_response(query, qa_chain):

    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']


def get_user_input():

    # get the user query
    input_text = st.text_input('Ask me anything about Promtior', "", key='input')
    return input_text



# create the qa_chain
qa_chain = create_qa_chain()


# create empty lists for user queries and responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# get the user query
user_input = get_user_input()


if user_input:

    # generate response to the user input
    response = generate_response(query=user_input, qa_chain=qa_chain)

    # add the input and response to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)


# display conversaion history (if there is one)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')