from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI,ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from ..config import Config
from ..Data.pinecone_client import PineconeClient
from .document_service import DocumentService
import os
import logging
import time
import openai

openai.api_key = Config.OPENAI_API_KEY


class AnswerGeneration:
    def __init__(self) -> None:
        self.pinecone_client = PineconeClient()
        self.doc_service = DocumentService()
        self.llm = self._initialize_llm()
        self.knowledgebase = self._initialize_vectorstore(Config.PINECONE_KNOWLEDGEBASE_NAME)
        self.guidelines = self._initialize_vectorstore(Config.PINECONE_GUIDELINES_NAME)

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the OpenAI Chat LLM."""
        try:
            return ChatOpenAI(
                model="gpt-4o", 
                temperature=0)
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            return None

    def _initialize_vectorstore(self, namespace: str) -> Pinecone:
        """Initialize the Pinecone vectorstore with the given namespace."""
        try:
            return Pinecone.from_existing_index(
                Config.PINECONE_INDEX_NAME,
                embedding=OpenAIEmbeddings(
                    model="text-embedding-ada-002",),

                namespace=namespace,
            )
        except Exception as e:
            logging.error(f"Error initializing vectorstore for namespace {namespace}: {e}")
            return None
    
    def create_question_extraction_pipeline(self, vectorstore, llm):
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )
        return qa_chain
    
    def answer_questions(self,qa_chain,input):
        query = f"""
        [INST] You are a question answering chatbot. Based on the provided context, first analyze the whole context of the document including headings and sub headings of each chunk and then answer the question. If the qustion does not have the context at all, answer from your knowledge.question:query: {input} [/INST]
        """
        result = qa_chain.invoke({"query": query})
        return result["result"]

    def extract_guidelines(self, qa_chain, input):
        query = f"""
        [INST] You are a guideline extraction chatbot. Based on the provided context, first analyze the whole context of the document and then extract the guidelines related to how to answer the question. Make it brief within 50 words. question:query: {input} [/INST]
        """

        result = qa_chain.invoke({"query": query})
        return result["result"]
    
    def answer_questions_with_guidelines(self, qa_chain,input,guideline):
        query = f"""
        [INST] You are a question answering extraction chatbot. Based on the provided context, analyze the entire document, including headings and subheadings, then answer the questions following the guidelines.
        Question: {input}
        Extracted Guidelines: {guideline}
        Provide a concise and formal answer. [/INST]"""

        result = qa_chain.invoke({"query": query})
        return result["result"]


    def generate_answers(self, questions: list[str]) -> list[str]:
        """Generate answers for the given questions using the LLM and vectorstore."""
        try:
            qa_chain_knowledgebase = self.create_question_extraction_pipeline(self.knowledgebase, self.llm)
            answers = []
            for question in questions:
                answer = self.answer_questions(qa_chain_knowledgebase, question)
                answers.append(answer)
            return answers
        except Exception as e:
            logging.error(f"Error generating answers: {e}")

    def generate_answers_using_guidelines(self, questions: list[str]) -> list[str]:
        """Generate answers for the given questions using the LLM and vectorstore."""
        try:
            qa_chain_guidelines = self.create_question_extraction_pipeline(self.guidelines, self.llm)
            extracted_guidelines = []
            extracted_answers = []
            qa_chain_knowledgebase = self.create_question_extraction_pipeline(self.knowledgebase, self.llm)
            for question in questions:
                extracted_guideline = self.extract_guidelines(qa_chain_guidelines, question)

                extracted_guidelines.append(extracted_guideline)
                answer = self.answer_questions_with_guidelines(qa_chain_knowledgebase, question, extracted_guideline)
                print(f"Question: {question}")
                print("-" * 40)
                print(f"Extracted Answer: {answer}")
                print("-" * 40)
                extracted_answers.append(answer)
            # answers = []
            # for question in questions:
            #     answer = answer_questions(qa_chain, question)
            #     answers.append(answer)
            return extracted_answers
        except Exception as e:
            logging.error(f"Error generating answers: {e}")
        

async def main():
    doc_service = DocumentService()
    pinecone_client = PineconeClient()
    answer_generation = AnswerGeneration()
    # For embedding and then storing the vector embeddings
    
    index = await pinecone_client.create_index()
    knowledge_docs = doc_service.load_documents("docs/financial_data/BSBFIN501 Student Guide.docx")
    knowledge_docs_chunks = doc_service.split_documents(knowledge_docs)
    vectors = await doc_service.generate_embeddings(knowledge_docs_chunks)
    # print(vectors)
    await pinecone_client.upsert_embeddings(vectors, index, namespace="knowledgebase")

    knowledge_docs_chunks = doc_service.split_documents(knowledge_docs)
    guidelines_docs = doc_service.load_documents("docs/financial_data/BSBFIN501 Assessor Marking Guide.docx")
    guidelines_docs_chunks = doc_service.split_documents(guidelines_docs)
    guideline_vectors = await doc_service.generate_embeddings(guidelines_docs_chunks)
    await pinecone_client.upsert_embeddings(guideline_vectors, index, namespace="guidelines")

    # for retreiving answers

    # answer_generation = AnswerGeneration()
    # answers = answer_generation.generate_answers(["Explain the basic principle of double entry bookkeeping?"])

    # questions = doc_service.read_questions("questions.txt")
    # answers_guidelines = answer_generation.generate_answers_using_guidelines(questions)
    # doc_service.save_to_file(questions, answers_guidelines, "answers_with_guidelines.txt")
    # answers = answer_generation.generate_answers(questions)
    # doc_service.save_to_file(questions, answers, "answers.txt")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())