from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pymilvus import DataType, MilvusClient, MilvusException
class VectorDataBaseServices:
    def __init__ (
        self,
        uri: str = os.getenv("MILVUS_URI"),
        token: str = os.getenv("MILVUS_TOKEN")
    ) -> None:
        self._uri = uri
        """Database remote connection settings"""
        self._token = token
        """Database security access token"""
        self._embeddings_model = HuggingFaceEmbeddings()
        """Model to use for embeddings"""
        self._client = MilvusClient(uri=self._uri, token=self._token)

        self._collection_name = os.getenv("MILVUS_COLLECTION")

        
    def create_records(self, collection_name: str, filepath: str) -> dict | MilvusException:
        data = []
        loader = PyMuPDFLoader(filepath)
        pages = loader.load_and_split()
        documents_content = [page.page_content for page in pages]
        print("Creando embeddings")
        embeddings = self._embeddings_model.embed_documents(documents_content)
        for i, vector in enumerate(embeddings):
            data.append({"vector": vector, "text": documents_content[i]})
        print("Guardando embeddings")
        try:
            response = self._client.insert(collection_name=collection_name, data=data)
            response["ids"] = list(response["ids"])
            return response
        except MilvusException as e:
            return e

    def create_collection(self, collection_name: str) -> str | MilvusException:
        try:
            schema = self._client.create_schema(enable_dynamic_field=True, description="Costarican criminal code vectors")
            schema.add_field(field_name="id", datatype=DataType.INT64, description="The Primary Key", is_primary=True, auto_id=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, description="Vector of document chunk", dim=768)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, description="Text of document chunk", max_length=65535)
            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="L2", index_type="IVF_FLAT")
            self._client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        except MilvusException as e:
            return e
        return "Collection successfully created"

    def similarity_search(self, query: str) -> list[list[dict]] | MilvusException:
        query_vector = self._embeddings_model.embed_query(query)
        try:
            response = self._client.search(self._collection_name, data=[query_vector], limit=3, output_fields=["text"], search_params={"metric_type": "L2", "params": {"range_filter": 1.0}})
            content = [response['entity']['text'] for response in response[0]]
            return content
        except MilvusException as error:
            return error

def get_vectorstore(uri: str = os.getenv("MILVUS_URI"), token: str = os.getenv("MILVUS_TOKEN")) -> VectorDataBaseServices:
    return VectorDataBaseServices(uri, token)
