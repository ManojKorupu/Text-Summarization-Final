import os
import spacy
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from PyPDF2 import PdfReader
import docx

app = Flask(__name__)

# ===== CONFIG (GROQ FINAL) =====
API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None

stored_text = ""

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ===== HELPERS =====

def check_api():
    if not API_KEY or not client:
        return False
    return True

def extract_text(file):
    try:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file)
            return " ".join([page.extract_text() or "" for page in reader.pages])
        elif file.filename.endswith(".docx"):
            doc = docx.Document(file)
            return " ".join([para.text for para in doc.paragraphs])
        elif file.filename.endswith(".txt"):
            return file.read().decode("utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return ""

def extract_entities(text):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def highlight_entities(text, entities):
    # Avoid duplicate replacements
    unique_entities = sorted(set([e["text"] for e in entities]), key=len, reverse=True)

    for entity in unique_entities:
        text = text.replace(entity, f"<mark>{entity}</mark>")
    return text

# ===== ROUTES =====

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global stored_text

    if not check_api():
        return jsonify({"error": "API key not configured. Set GROQ_API_KEY."})

    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        stored_text = extract_text(file)
    else:
        stored_text = request.form.get("text", "")

    if not stored_text.strip():
        return jsonify({"error": "No document provided!"})

    return jsonify({"message": "Document uploaded successfully!"})

@app.route("/summarize", methods=["POST"])
def summarize():
    if not check_api():
        return jsonify({"error": "API key not configured. Set GROQ_API_KEY."})

    if not stored_text:
        return jsonify({"error": "Upload document first!"})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize clearly and professionally."},
                {"role": "user", "content": stored_text}
            ]
        )

        if not response or not response.choices:
            return jsonify({"error": "No response from model."})

        summary = response.choices[0].message.content.strip()

        entities = extract_entities(summary)
        highlighted_summary = highlight_entities(summary, entities)

        return jsonify({
            "summary": highlighted_summary,
            "entities": entities
        })

    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"})

@app.route("/ask", methods=["POST"])
def ask():
    if not check_api():
        return jsonify({"error": "API key not configured. Set GROQ_API_KEY."})

    if not stored_text:
        return jsonify({"error": "Upload document first!"})

    question = request.json.get("question", "")

    if not question.strip():
        return jsonify({"error": "Please enter a question."})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Answer ONLY using the provided document."},
                {"role": "user", "content": f"Document:\n{stored_text}\n\nQuestion:\n{question}"}
            ]
        )

        if not response or not response.choices:
            return jsonify({"error": "No response from model."})

        answer = response.choices[0].message.content.strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"Q&A failed: {str(e)}"})

# ===== RUN =====

if __name__ == "__main__":
    app.run(debug=True)