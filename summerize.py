from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
from createTranscript import createTranscript
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


def summerize(numOfSpeakers, videoPath):
    
    model_name = 'philschmid/bart-large-cnn-samsum'

    # Ensure CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load model and tokenizer
    original_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,  # Use bfloat16 if your GPU supports it
        device_map="auto"  # Automatically map model layers to available devices
    ).to(device)  # Move the model to GPU

    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

    prompt = createTranscript(numOfSpeakers, videoPath)

    # Tokenize input and move it to GPU
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    output = tokenizer.decode(
        original_model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=500,  # Generate up to 500 tokens
        no_repeat_ngram_size=3,  # Avoid repetition of 3-grams
        temperature=0.7,         # Control randomness (lower values are more deterministic)
        top_p=0.9,               # Use nucleus sampling for diverse outputs
        )[0], 
        skip_special_tokens=True
    )

    dash_line = '-' * 100
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')
    
    return output



app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    return jsonify({"message": "File uploaded successfully", "file_path": file_path})

@app.route("/summerize", methods=["POST"])
def generate_code():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    
    videoPath = data.get("videoPath", "")
    print(data.get("videoPath", ""))
    print(data.get("numOfSpeakers", ""))
    numOfSpeakers = data.get("numOfSpeakers", "")
    
    if not videoPath:
        print("No video Path received")
        return jsonify({"error": "Video path is required"}), 400
    
    summerizied = summerize(numOfSpeakers, videoPath)
    
    return jsonify({"summerized": summerizied})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    
    
