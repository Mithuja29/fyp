import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# user input preprocess code
def preprocess_code(code):
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)  # Remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Remove multi-line comments
    return code

# Load the trained lstm model
model = load_model("java_error_detection_model.h5")

# Load  dataset and fit the tokenizer
csv_filename = "Code_Dataset/java_errors_dataset.csv"
df = pd.read_csv(csv_filename)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['sampleCode'])
word_index = tokenizer.word_index
maxlen = max(len(seq) for seq in tokenizer.texts_to_sequences(df['sampleCode']))

# Error type mapping
error_types = df['typeofError'].unique()
error_type_to_index = {error_type: idx for idx, error_type in enumerate(error_types)}

def predict_error_type(model, code, tokenizer, maxlen, error_type_to_index):
    code = preprocess_code(code)
    sequence = tokenizer.texts_to_sequences([code])
    input_data = pad_sequences(sequence, maxlen=maxlen, padding='post')
    prediction = model.predict(input_data)
    error_index = np.argmax(prediction, axis=1)[0]
    error_type = list(error_type_to_index.keys())[list(error_type_to_index.values()).index(error_index)]
    return error_type

@app.route('/')
def index():
    return render_template('Detect_UI.html')

@app.route('/predict_Error', methods=['POST'])
def predict_Error():
    code = request.form['code']

    def is_valid_java_code(code):
        # Define regex patterns
        class_pattern = r'\b(public|private|protected)\s+class\b'
        method_pattern = r'\b(public|private|protected|static|final|\s)\s+\b(class|void|int|double|float|long|short|byte|char)\s+[a-zA-Z_$][a-zA-Z\d_$]*\s*\([^)]*\)\s*[{;]'
        main_method_pattern = r'\b(public|private|protected|static)\s+void\s+main\s*\(\s*String\s*\[\s*\]\s*[a-zA-Z_$][a-zA-Z\d_$]*\s*\)\s*[{;]'

        # Check if  patterns match
        if re.search(class_pattern, code, re.MULTILINE) and re.search(method_pattern, code, re.MULTILINE) and re.search(
                main_method_pattern, code, re.MULTILINE):
            return True
        else:
            return False

    code_snippet = code

    # out put to user
    if is_valid_java_code(code_snippet):
        # error type prediction to UI
        error_type = predict_error_type(model, code, tokenizer, maxlen, error_type_to_index)
        return jsonify({'error_type': error_type})
    else:
        # invalid java code message to UI
        error_type = "The code snippet is not valid Java code."
        return jsonify({'error_type': error_type})

if __name__ == '__main__':
    app.run(debug=True)
