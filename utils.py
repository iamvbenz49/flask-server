from transformers import AutoTokenizer, AutoModel
import torch, os
from pymilvus import MilvusClient


model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
url = os.getenv("URI")
client = MilvusClient(uri=url)

def embedding_fn(queries):   
    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)  
    return embeddings.numpy()


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
