from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from main import ask 

app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"service": "groq-serpapi-chatbot", "endpoints": ["/health", "/api/chat"]})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Provide 'message' (non-empty string) in JSON body"}), 400

    try:
        answer = ask(message.strip())
        return jsonify({"answer": answer})
    except Exception as e:
        # In production, log traceback here
        return jsonify({"error": str(e)}), 500

# Optional: nicer JSON for common HTTP errors
@app.errorhandler(HTTPException)
def handle_http_exc(err: HTTPException):
    return jsonify({"error": err.name, "status_code": err.code}), err.code

if __name__ == "__main__":
    # For local dev only. Use gunicorn in prod: gunicorn -w 2 -b 0.0.0.0:8000 app:app
    app.run(host="0.0.0.0", port=8000, debug=True)
