from dotenv import load_dotenv
import os

load_dotenv()
class Config:
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING = os.environ.get("AZURE_OPENAI_EMBEDDING")
    AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT_NAME")
