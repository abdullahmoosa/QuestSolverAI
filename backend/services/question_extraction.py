import nltk
import time
import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from langchain.chains import RetrievalQA


nltk.download("punkt")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=2500)
    chunks = text_splitter.split_documents(documents)
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING"),
        ),
        collection_name="local-rag",
    )

    return vector_db


def setup_llm():
    # local_model = "llama3.1:latest"
    llm = AzureChatOpenAI(
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    return llm


def create_question_extraction_pipeline(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain


def extract_questions(qa_chain):
    query = """
    [INST] Based on the content of the document, find all the questions for assesment task 1. Make sure you extract all the questions. Format your response as a numbered list and if found nested then use bulleted list for the nested items.
    [/INST]
    """
    result = qa_chain.invoke({"query": query})
    questions = result["result"]

    # Save the questions to a text file
    with open("questions.txt", "w") as file:
        file.write(questions)

    return questions


def main():
    file_path = "docs/financial_data/BSBFIN501 Student Assessment Tasks.docx"

    documents = load_docx(file_path)
    splits = split_documents(documents)
    vectorstore = create_vector_store(splits)
    llm = setup_llm()

    start_time = time.time()
    qa_chain = create_question_extraction_pipeline(vectorstore, llm)

    questions = extract_questions(qa_chain)
    end_time = time.time()
    print("Extracted Questions:")
    print(questions)
    print(f"Questions saved to 'questions.txt'")
    print(f"Time spent: {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
