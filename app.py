from flask import Flask, request, jsonify, Response
from langchain_google_genai import ChatGoogleGenerativeAI
import os, requests, time
from dotenv import load_dotenv
from pymilvus import MilvusClient, model


load_dotenv()

client = MilvusClient("milvus_demo.db")

embedding_fn = model.DefaultEmbeddingFunction()

app = Flask(__name__)
def fetch_documets(query):
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
        yield f"{word} "  # Send the word followed by a space
        time.sleep(0.05)  # Delay for demonstration (50ms)
@app.route("/query", methods=["POST"])
def query_response():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"Received query: {query}")

    try:
        llm = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        relevant_docs = fetch_documets(query)
        print(relevant_docs)
        query = "These are the relevant documents, " + str(relevant_docs) + "Based on this answer the query" + query
        res = llm.invoke(query)
        
        content = res.content if hasattr(res, 'content') else "No content available" 
        

        return Response(generate(content), content_type='text/plain', headers={'Transfer-Encoding': 'chunked'})

    except requests.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return jsonify({"error": "Failed to connect to Gemini API"}), 500

if __name__ == "__main__":
    fetch_documets("")
    app.run(debug=True)
