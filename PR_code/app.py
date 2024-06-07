import os
import argparse
import yaml
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from utils.image_processing import preprocess_and_extract_features
from utils.text_processing import load_tokenizer, generate_caption

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load configuration
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Load model and tokenizer
model = load_model(config['model_save_path'])
tokenizer = load_tokenizer(config['tokenizer_path'])
max_length = config['max_length']

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and generate caption
        image = preprocess_and_extract_features([filepath])[0]
        caption = generate_caption(model, tokenizer, image, max_length)
        
        return render_template('result.html', caption=caption, image_url=filename)

if __name__ == '__main__':
    app.run(debug=True)
