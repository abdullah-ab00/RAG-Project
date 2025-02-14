import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os


load_dotenv()
gemini_key = os.getenv('gemini_key')

st.markdown("""
    <h1 style="text-align: center;">RAG BOT</h1>
""", unsafe_allow_html=True)
st.markdown("""
### Follow these simple steps to interact with the chatbot:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=gemini_key)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_embeddings(text_chunks, file_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
    embedding_dir = f"embeddings_{file_name}"
    
    if os.path.exists(embedding_dir):
        print("Loading existing embeddings...")
        return FAISS.load_local(embedding_dir, embeddings, allow_dangerous_deserialization=True)
      
    print("Making new embeddings...")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) 
    vector_store.save_local(embedding_dir)
    return FAISS.load_local(embedding_dir, embeddings, allow_dangerous_deserialization=True)


pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
vec_embd = None  

if pdf_docs:
    with st.spinner("Processing..."):
        extracted_text = get_pdf_text(pdf_docs)
        chunks = get_text_chunks(extracted_text)
        filename = pdf_docs[0].name
        vec_embd = get_vector_embeddings(chunks, filename)  
        st.success("Done")

user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

if vec_embd and user_question:
    docs = vec_embd.similarity_search(user_question)

    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "Answer is not available in the Context". 
    Don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}?\n
    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"context": docs, 'question': user_question})
    st.write("Reply: ", response)
else:
    if not pdf_docs:
        st.warning("Please upload PDF file to enable Questioning.")
    elif not user_question:
        st.warning("Please enter a Question.")

