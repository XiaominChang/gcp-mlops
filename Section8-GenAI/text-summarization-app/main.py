"""
Section 8 - Lecture 7: Document Summarization Application on Cloud Run
Flask application using Gemini 2.0 Flash for Word document summarization.

Reads .docx files from a GCS bucket and returns a concise summary.

Endpoints:
    POST /summarize_word_documents - Summarize a Word document from GCS
    GET  /health                   - Health check
"""

import os
from io import BytesIO

import vertexai
from docx import Document
from flask import Flask, request, jsonify
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, GenerationConfig

app = Flask(__name__)

# Initialize Vertex AI
vertexai.init()

# Create a Gemini model instance
model = GenerativeModel("gemini-2.0-flash")

# GCS client and bucket
client = storage.Client()
BUCKET_NAME = "sid-ml-ops"
bucket = client.bucket(BUCKET_NAME)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "gemini-2.0-flash"})


@app.route("/summarize_word_documents", methods=["POST"])
def summarize_word_documents():
    """Read a Word document from GCS and return a Gemini-generated summary."""
    input_request_json = request.get_json()
    if not input_request_json or "file_name" not in input_request_json:
        return jsonify({"error": "Missing 'file_name' field in request body"}), 400

    doc_file = input_request_json["file_name"]

    try:
        # Read the Word document from GCS
        blob = bucket.blob("gen-ai/" + doc_file)
        word_file_in_bytes = blob.download_as_bytes()
        document = Document(BytesIO(word_file_in_bytes))

        # Extract all paragraph text
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)
        full_text = "\n".join(text)

        if not full_text.strip():
            return jsonify({"error": "Document appears to be empty"}), 400

        # Generate summary with Gemini
        prompt = f"""Provide a very short summary, no more than 50 words, for the following article:
text: {full_text}
"""
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))
