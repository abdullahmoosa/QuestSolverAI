from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from ..config import Config
import os
import logging
import time
def load_docx(file_path):
    try:
        if os.path.exists(file_path):
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents")
            return documents
        # return loader.load()
        else:
            logging.info(f"File not found: {file_path}")
            return []
    except Exception as e:
        logging.error(f"Error loading document: {e}")


def split_documents(documents):
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=AzureOpenAIEmbeddings(
                model="text-embedding-ada-002",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                api_key=Config.AZURE_OPENAI_API_KEY,
                azure_deployment=Config.AZURE_OPENAI_EMBEDDING,
            ),
        collection_name="local-rag",
    )

    return vector_db

def create_question_extraction_pipeline(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain

def setup_llm():
    # local_model = "llama3.1:latest"
    llm = AzureChatOpenAI(
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            openai_api_version=Config.AZURE_OPENAI_API_VERSION,
            openai_api_key=Config.AZURE_OPENAI_API_KEY,
            temperature= 0.00001
        )
    return llm

def answer_questions(qa_chain,input):
    query = f"""
    [INST] Based on the provided context, answer the question. query: {input} [/INST]
    """
    result = qa_chain.invoke({"query": query})
    return result["result"]

def read_questions(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    questions = []
    current_question = ""

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace

        # Check if the line starts with a number followed by a period, indicating a new question
        if line and line[0].isdigit() and line[1] == '.':
            if current_question:  # If there's a current question, save it
                questions.append(current_question)
            current_question = line  # Start a new question
        else:
            # Append to the current question if it's a continuation
            if current_question:
                current_question += " " + line

    # Append the last question
    if current_question:
        questions.append(current_question)

    return questions

    return questions

def read_and_clean_questions(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    questions = []
    current_question = ""

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace

        # Check if the line starts with a number followed by a period, indicating a new question
        if line and line[0].isdigit() and line[1] == '.':
            if current_question:  # If there's a current question, save it
                questions.append(current_question)
            current_question = line  # Start a new question
        else:
            # Append to the current question if it's a continuation
            if current_question:
                current_question += " " + line

    # Append the last question
    if current_question:
        questions.append(current_question)

    # Remove numbering from each question
    cleaned_questions = [q.split('. ', 1)[-1] for q in questions]

    return cleaned_questions

def main():
    questions = read_questions("questions.txt")
    cleaned_questions = read_and_clean_questions("questions.txt")
    print(cleaned_questions)
    print("------------------------")

    file_path = "/Users/relisource/Personal Projects/QuestSolver/docs/BSBFIN501 Student Guide.docx"

    documents = load_docx(file_path)
    # print(f"documents : {documents}")
    splits = split_documents(documents)
    vectorstore = create_vector_store(splits)
    llm = setup_llm()  # need to use opeanai here

    qa_chain = create_question_extraction_pipeline(vectorstore, llm)
    print("----------------------------------")
    print("\n")
    for index, question in enumerate(cleaned_questions):
        start_time = time.time()
        answer = answer_questions(qa_chain,question)
        end_time = time.time()
        print(f"question {index+1}: {question}")
        print(f"answer : {answer}")
        print(f"Time spent: {end_time-start_time} seconds")
        print("--------------------------------")
        print("\n")
    # answer = answer_questions(qa_chain,"List three types of budgets.")
    # print(f"answer : {answer}")
    # end_time = time.time()
    # print("Extracted Questions:")
    # print(questions)

if __name__ == "__main__":
    main()