from pymilvus import model
from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "Metaphysics: Studies the nature of reality, existence, and the universe.",
    "Epistemology: Focuses on knowledge, belief, and the nature of truth.",
    "Ethics (Moral Philosophy): Concerns questions of right and wrong, good and evil.",
    "Aesthetics: Examines beauty, art, and taste.",
    "Logic: Studies the principles of valid reasoning and argument.",
    "Political Philosophy: Explores questions about government, justice, rights, and the role of individuals in society.",
    "Philosophy of Mind: Investigates the nature of the mind, consciousness, and their relationship to the body.",
    "Philosophy of Language: Examines the nature of language, its relations to reality, and how we communicate.",
    "Philosophy of Science: Analyzes the methods and implications of science.",
    "Existentialism: Focuses on individual existence, freedom, and choice.",
    "Pragmatism: Emphasizes practical consequences and real effects as the criteria for meaning and truth.",
    "Phenomenology: Studies structures of experience and consciousness.",
    "Structuralism: Analyzes cultural phenomena in terms of their relationship to a larger, overarching system.",
    "Postmodernism: Challenges traditional narratives and ideologies, emphasizing relativism and skepticism."
]




vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="demo_collection", data=data)

print(res)
