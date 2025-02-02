from pymilvus import model
import os, requests
from pymilvus import MilvusClient


url = os.getenv("URI")
embedding_url = os.getenv("EMBEDDING_URL")
hf_token = os.getenv("HF_TOKEN")
milvus_uri = os.getenv("MILVUS_URI")
milvus_password = os.getenv("MILVUS_PASSWORD")

client = MilvusClient(
    uri=milvus_uri,
    token=milvus_password
)

model_id = "sentence-transformers/all-MiniLM-L6-v2"



def embedding_fn(texts):   
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()



def fetch_documents(query, collection_name = "marveldcc"):
    query_vectors = embedding_fn([query])
    res = client.search(
        collection_name=collection_name, 
        data=query_vectors,  
        limit=2, 
        output_fields=["text"],
    )
    docs = [doc["entity"]["text"] for doc in res[0]]
    return docs





if __name__ == '__main__':
    queries = ["Who is Alan Turing?", "What is AI?"]
    query_vectors = embedding_fn(queries)
    print(hf_token)
    print(query_vectors)
