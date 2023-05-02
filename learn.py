from flask import Flask, render_template, request, jsonify
import fitz
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

app = Flask(__name__)

# Step 1: Choose a pre-trained model architecture
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Step 2: Install the required libraries and packages
# Already installed in the virtual environment

# Step 3: Load the pre-trained model
try:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except OSError:
    print(f"Model not found locally. Downloading {model_name}...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 4: Define a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        return jsonify({'message': 'File uploaded successfully.'}), 200
    else:
        return jsonify({'error': 'No file selected.'}), 400

@app.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']
    filepath = request.form['filepath']

    # Parse the uploaded PDF document
    doc = fitz.open(filepath)
    text = ""
    if doc.page_count > 0:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
    doc.close()

    # Use the pre-trained BERT model to extract answers
    inputs = tokenizer.encode_plus(question, text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax(dim=-1).item()
    answer_end = outputs.end_logits.argmax(dim=-1).item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1]))

    # Render the results on a separate page
    return render_template('answer.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)