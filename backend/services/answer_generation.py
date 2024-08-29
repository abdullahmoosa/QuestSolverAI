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
    result = qa_chain({"query": query})
    return result["result"]

def main():
    file_path = "/Users/relisource/Personal Projects/QuestSolver/docs/BSBFIN501 Student Assessment Tasks.docx"

    documents = load_docx(file_path)
    # print(f"documents : {documents}")
    splits = split_documents(documents)
    vectorstore = create_vector_store(splits)
    llm = setup_llm()  # need to use opeanai here

    start_time = time.time()
    qa_chain = create_question_extraction_pipeline(vectorstore, llm)
    answer = answer_questions(qa_chain,"List three types of budgets.")
    print(f"answer : {answer}")
    end_time = time.time()
    # print("Extracted Questions:")
    # print(questions)
    print(f"Time spent: {end_time-start_time} seconds")


if __name__ == "__main__":
    main()