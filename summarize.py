from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
from createTranscript import createTranscript
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


def summarize(numOfSpeakers, videoPath):
    
    model_name = 'philschmid/bart-large-cnn-samsum'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

    prompt = createTranscript(numOfSpeakers, videoPath)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    output = tokenizer.decode(
        original_model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=500,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9,
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
        return jsonify({"error": "No file path"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    return jsonify({"message": "File uploaded successfully", "file_path": file_path})

@app.route("/summarize", methods=["POST"])
def summarizeVideo():
    
    if "file_path" not in request.form:
        return jsonify({"error": "No file path"}), 400
    
    if "num_of_speakers" not in request.form:
        return jsonify({"error": "No number of speakers"}), 400
    
    videoPath = request.form["file_path"]
    numOfSpeakers = request.form["num_of_speakers"]
    
    if not videoPath:
        return jsonify({"error": "Video path is required"}), 400

    summarized = summarize(numOfSpeakers, videoPath)
    
    return jsonify({"summarized": summarized})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    
    
