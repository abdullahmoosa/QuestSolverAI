import os
import logging
import openai
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from ..config import Config

openai.api_key = Config.OPENAI_API_KEY
class DocumentService:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002")
    #     self.embeddings = AzureOpenAIEmbeddings(
    #         model=Config.AZURE_OPENAI_EMBEDDING,
    #         azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    #         api_version=Config.AZURE_OPENAI_API_VERSION,
    #         api_key=Config.AZURE_OPENAI_API_KEY,
    #         azure_deployment=Config.AZURE_OPENAI_EMBEDDING,
    # )

                
    def load_documents(self, file_path):
        try:
            if not os.path.exists(file_path):
                logging.info(f"File not found: {file_path}")
                return []
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.pdf':
                loader = UnstructuredPDFLoader(file_path)
            else:
                logging.info(f"Unsupported file type: {file_extension}")
                return []

            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logging.error(f"Error loading document: {e}")
            return []
        
    def split_documents(self, documents, chunk_size=3000, chunk_overlap=1500):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    async def generate_embeddings(self, chunks):
        """Method to generate embeddings from chunks."""
        vector_embeddings = []
        id = 0
        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk.page_content)
            id = id + 1
            chunk.metadata["text"] = chunk.page_content
            structured_vector = {
                "id": str(id),
                "values": embedding,
                "metadata": chunk.metadata,
            }
            vector_embeddings.append(structured_vector)
        assert len(vector_embeddings) > 0
        return vector_embeddings
    
    
    def read_questions(self, file_path):
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

    def save_to_file(self,questions, answers, output_file):
        with open(output_file, "w") as file:
            for index, (question, answer) in enumerate(zip(questions, answers)):
                file.write(f"{question}\n")
                file.write(f"Answer: {answer}\n")
                # file.write(f"Time spent: {answer['time_spent']} seconds\n")
                file.write("-" * 40 + "\n\n")