import logging
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from ..config import Config

logging.basicConfig(format="%(filename)s: %(message)s", level=logging.INFO)

class PineconeClient:
    """Class to interact with Pinecone."""

    def __init__(self):
        self.pinecone = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME

    async def create_index(self, name=None, metric="cosine"):
        """Method to create an index in Pinecone."""
        name = name or self.index_name
        if name not in self.pinecone.list_indexes().names():
            logging.info(f"Index '{name}' not found, creating a new index.")
            self.pinecone.create_index(
                name=name,
                dimension=1536,  # Adjust this based on your embeddings
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        index = self.pinecone.Index(name=name)
        return index

    async def upsert_embeddings(self, vector_embeddings, index, namespace):
        """Method to upsert vectors in Pinecone."""
        try:
            index.upsert(vectors=vector_embeddings, namespace=namespace)
            logging.info(f"Upserted {len(vector_embeddings)} vectors into namespace '{namespace}'.")
        except Exception as e:
            logging.error(f"Error during upsert: {e}")
        return "Vectors upserted."

    async def delete_a_namespace(self, doc_names, index):
        """Method to delete a namespace in Pinecone."""
        for name in doc_names:
            try:
                index.delete(namespace=name, delete_all=True)
                logging.info(f"Deleted namespace: {name}")
            except Exception as e:
                logging.error(f"Error deleting namespace '{name}': {e}")
        return "Namespaces deleted."

    async def delete_vectors_per_document(self, doc_name: str, index, namespace: str):
        """Method to delete stored vectors for a specific document in Pinecone."""
        assert doc_name is not None
        delete_ids = index.list(prefix=doc_name, namespace=namespace)
        try:
            index.delete(ids=delete_ids, namespace=namespace)
            logging.info(f"Deleted vectors for document: {doc_name} in namespace: {namespace}")
        except Exception as e:
            logging.error(f"Error deleting vectors for document '{doc_name}': {e}")
        return "Vectors deleted."
