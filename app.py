from flask import Flask, request, jsonify, Response
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from utils import fetch_documents
import time, os

app = Flask(__name__)


def generate(words):
    split_words=words.split(" ")
    for word in split_words:
        yield f"{word} "
        time.sleep(0.05)


@app.route("/retrieve", methods=["POST"])
def query_response():
    data = request.json
    query = data.get("question")

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
        query = f"Generate a title (5 words max) for: {query}"
        res = llm.invoke(query)
        content = res.content if hasattr(res, 'content') else "No content available"

        return jsonify({"title": content}), 200

    except requests.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return jsonify({"error": "Failed to connect to Gemini API"}), 500
if __name__ == "__main__":
    print("hello")
    app.run(debug=True)
