from dotenv import load_dotenv
import os

load_dotenv()
class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_KNOWLEDGEBASE_NAME = "knowledgebase"
    PINECONE_GUIDELINES_NAME = "guidelines"
    PINECONE_PROJECT_GUIDELINES_NAME = "project_guidelines"
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
