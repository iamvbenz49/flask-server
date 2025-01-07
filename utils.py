from pymilvus import model
import os, requests
from pymilvus import MilvusClient


url = os.getenv("URI")
embedding_url = os.getenv("EMBEDDING_URL")
hf_token = os.getenv("HF_TOKEN")
client = MilvusClient(uri=url)


def embedding_fn(queries):   
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = '{"inputs": "Deploying my first endpoint was an amazing experience."}'
    response = requests.post(embedding_url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"


def fetch_documents(query, collection_name = "demo_collection"):
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

    print(query_vectors)
