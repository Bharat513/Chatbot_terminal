#import streamlit as st
import warnings
import time
import os
import glob
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


warnings.filterwarnings("ignore")


MODEL = "llama3"

model = Ollama(model=MODEL) 


## for extracting the files from the folder
def list_files_in_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    file_names = [os.path.basename(file) for file in files]
    return file_names



## extracting the text from the pdf
def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text       




# convert all text into multiple chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks




# with the help of OllamaEmbeddings, convert all the chunks into vector embedding and store that vector into FAISS Database
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model=MODEL)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore




# creating the conversational chain with the help of llama2 model
def get_conversational_chain():
    template = """
    Answer the question based on the context below. If you can't
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


# take the user input and send back an output according to the query
def user_input(user_question):
    embeddings = OllamaEmbeddings(model=MODEL)
    
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization = True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True
    )

    print(response)
    #st.write("Reply: ", response["output_text"])












def main():
    # for calling all functions that will take input from the folder and create and storing embedding 
    # folder_path = 'pdfs' 
    # files = list_files_in_folder(folder_path)
    # print("Files in folder:")

    # pdf_docs = [os.path.join(folder_path, file) for file in files]  
    # for file in pdf_docs:
    #     print(file)
    #     text = get_pdf_text([file]) 

    #     print("converted into text")
    #     start_time = time.time()
    #     chunks = get_text_chunks(text)
    #     end_time = time.time()
    #     print("converted into chunks")
    #     elapsed_time = end_time - start_time
    #     print("Elapsed Time in creating chunks:", elapsed_time, "seconds")
    #     start_time = time.time()
    #     vectorstore = get_vectorstore(chunks)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print("stored into database")
    #     print("Elapsed Time in creating embeddings and store in faiss:", elapsed_time, "seconds")



    user_question = input("Please enter your question: ")
    start_time = time.time()
    if user_question:
        user_input(user_question)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time in generating the response for a user query:", elapsed_time, "seconds")




if __name__ == '__main__':
    main()


