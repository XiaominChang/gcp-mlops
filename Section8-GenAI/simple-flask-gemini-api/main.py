"""
Section 8 - Lecture 6: Text Classification LLM on Cloud Run
Flask application using Gemini 2.0 Flash for text toxicity classification.

Endpoints:
    POST /simple_classification        - Classify text as toxic or non-toxic
    POST /simple_classification_with_exp - Classify with explanation
    GET  /health                        - Health check
"""

import os
import vertexai
from flask import Flask, request, jsonify
from vertexai.generative_models import GenerativeModel, GenerationConfig

app = Flask(__name__)

# Initialize Vertex AI - project ID is auto-detected on Cloud Run
vertexai.init()

# Create a Gemini model instance
model = GenerativeModel("gemini-2.0-flash")

# Shared generation config for classification tasks
classification_config = GenerationConfig(
    temperature=0.1,
    max_output_tokens=256,
)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "gemini-2.0-flash"}


@app.route("/simple_classification", methods=["POST"])
def simple_classification():
    """Classify text as toxic or non-toxic."""
    input_request_json = request.get_json()
    if not input_request_json or "msg" not in input_request_json:
        return jsonify({"error": "Missing 'msg' field in request body"}), 400

    input_txt = input_request_json["msg"]
    prompt = f"""Given a piece of text, classify it as toxic or non-toxic.
text: {input_txt}
"""
    response = model.generate_content(
        prompt,
        generation_config=classification_config,
    )
    return jsonify({"response": response.text})


@app.route("/simple_classification_with_exp", methods=["POST"])
def classification_with_exp():
    """Classify text as toxic or non-toxic with an explanation."""
    input_request_json = request.get_json()
    if not input_request_json or "msg" not in input_request_json:
        return jsonify({"error": "Missing 'msg' field in request body"}), 400

    input_txt = input_request_json["msg"]
    prompt = f"""Given a piece of text, classify it as toxic or non-toxic and explain why.
text: {input_txt}
"""
    response = model.generate_content(
        prompt,
        generation_config=classification_config,
    )
    return jsonify({"response": response.text})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5052)))
