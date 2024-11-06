from flask import Flask, request, jsonify, Response
from langchain_google_genai import ChatGoogleGenerativeAI
import os, requests, time
from dotenv import load_dotenv
from pymilvus import MilvusClient, model

load_dotenv()

# Initialize Milvus client and setup collection
client = MilvusClient("milvus_demo.db")
embedding_fn = model.DefaultEmbeddingFunction()

app = Flask(__name__)

@app.route("/insert", methods=["POST"])
def setup_insert():
    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")
    
    client.create_collection(
        collection_name="demo_collection",
        dimension=768,
    )

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
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
    ]

    client.insert(collection_name="demo_collection", data=data)
    print("Data inserted successfully")
    

    return jsonify({"message": "Data inserted successfully"}), 200



def fetch_documents(query):
    query_vectors = embedding_fn.encode_queries([query])
    res = client.search(
        collection_name="demo_collection", 
        data=query_vectors,  
        limit=2, 
        output_fields=["text"],
    )
    docs = [doc["entity"]["text"] for doc in res[0]]
    return docs


def generate(words):
    for word in words:
        yield f"{word} "
        time.sleep(0.05)

@app.route("/query", methods=["POST"])
def query_response():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        llm = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        relevant_docs = fetch_documents(query)
        query = f"These are the relevant documents: {relevant_docs}. Based on this, answer the query: {query}"
        res = llm.invoke(query)
        content = res.content if hasattr(res, 'content') else "No content available"

        return Response(generate(content), content_type='text/plain', headers={'Transfer-Encoding': 'chunked'})

    except requests.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return jsonify({"error": "Failed to connect to Gemini API"}), 500


@app.route("/title", methods=["POST"])
def query_title():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        llm = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        relevant_docs = fetch_documents(query)
        query = f"This is the query: {query}. Based on this, provide a title {query}"
        res = llm.invoke(query)
        content = res.content if hasattr(res, 'content') else "No content available"

        return jsonify({"title": content}), 200

    except requests.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return jsonify({"error": "Failed to connect to Gemini API"}), 500
if __name__ == "__main__":
    app.run(debug=True)
