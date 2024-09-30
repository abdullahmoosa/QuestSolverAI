from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=1500)
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
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature= 0,
    )
    return llm

def answer_questions(qa_chain,input):
    """
    Given a question and context from a document, use the question answer chain
    to generate an answer and its source context.

    Args:
        qa_chain: The question answer chain
        input: The question to ask

    Returns:
        A dictionary containing the answer and its source context
    """
    query = f"""
    [INST] You are a question answering chatbot. Based on the provided context, first analyze the whole context of the document including headings and sub headings of each chunk and then answer the question. Provide source of the answer.question: {input} [/INST]
    """
    result = qa_chain.invoke({"query": query})
    return result["result"]

def main():
    file_path = "/Users/relisource/Personal Projects/QuestSolver/docs/BSBFIN501 Student Guide.docx"

    documents = load_docx(file_path)
    # print(f"documents : {documents}")
    splits = split_documents(documents)
    vectorstore = create_vector_store(splits)
    llm = setup_llm()  # need to use opeanai here

    start_time = time.time()
    qa_chain = create_question_extraction_pipeline(vectorstore, llm)
    question = "List three key features of A New Tax System (GST) Act 1999."
    answer = answer_questions(qa_chain,question)
    print(f"question : {question}")
    print(f"answer : {answer}")
    end_time = time.time()
    print(f"Time spent: {end_time-start_time} seconds")


if __name__ == "__main__":
    main()