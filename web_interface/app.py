import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.logger_config import logger
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import io

from src.predict import load_vit, predict_image
from src.config import BASE_PATH, OUTPUT_PATH

app = Flask(__name__)

print('loading ViT model...')
logger.info("----------Loading ViT model----------")
feature_extractor, vit, device = load_vit()
logger.info('----------Ready!----------')

def save_prediction(image_name, results):
    """Save each prediction to predictions.json for Streamlit to read later"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    path = OUTPUT_PATH + '/predictions.json'

    if os.path.exists(path):
        with open(path, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append({
        'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image':        image_name,
        'manufacturer': {
            'label':      results['manufacturer']['label'],
            'confidence': round(results['manufacturer']['confidence'], 2)
        },
        'family': {
            'label':      results['family']['label'],
            'confidence': round(results['family']['confidence'], 2)
        },
        'variant': {
            'label':      results['variant']['label'],
            'confidence': round(results['variant']['confidence'], 2)
        }
    })

    with open(path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f'prediction saved -> {path}')

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Aircraft Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #0f172a; color: white; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
        h1 { font-size: 2rem; margin-bottom: 8px; }
        p.sub { color: #94a3b8; margin-bottom: 40px; }
        .card { background: #1e293b; border-radius: 16px; padding: 32px; width: 100%; max-width: 600px; }
        .upload-zone { border: 2px dashed #334155; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.2s; margin-bottom: 20px; }
        .upload-zone:hover { border-color: #3b82f6; background: #1e3a5f; }
        .upload-zone input { display: none; }
        .upload-zone label { cursor: pointer; color: #94a3b8; font-size: 1rem; }
        .upload-zone label span { color: #3b82f6; font-weight: bold; }
        #preview { width: 100%; border-radius: 8px; margin-bottom: 20px; display: none; max-height: 300px; object-fit: contain; }
        button { width: 100%; padding: 14px; background: #3b82f6; color: white; border: none; border-radius: 10px; font-size: 1rem; cursor: pointer; font-weight: bold; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        button:disabled { background: #334155; cursor: not-allowed; }
        .results { margin-top: 24px; display: none; }
        .result-item { background: #0f172a; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
        .result-item .level { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .result-item .label { font-size: 1.3rem; font-weight: bold; margin-bottom: 8px; }
        .result-item .confidence { color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px; }
        .bar-bg { background: #1e293b; border-radius: 999px; height: 8px; }
        .bar-fill { height: 8px; border-radius: 999px; background: #3b82f6; transition: width 0.5s; }
        .top3 { margin-top: 10px; }
        .top3-item { display: flex; justify-content: space-between; font-size: 0.85rem; color: #64748b; margin-bottom: 2px; }
        .loading { text-align: center; color: #94a3b8; padding: 20px; display: none; }
        .spinner { border: 3px solid #334155; border-top: 3px solid #3b82f6; border-radius: 50%; width: 30px; height: 30px; animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .manufacturer .bar-fill { background: #22c55e; }
        .family      .bar-fill { background: #3b82f6; }
        .variant     .bar-fill { background: #f97316; }
    </style>
</head>
<body>
    <h1>Aircraft Classifier</h1>
    <p class="sub">Upload an aircraft photo to identify it</p>

    <div class="card">
        <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            <label>Drop an image here or <span>browse</span></label>
        </div>

        <img id="preview" src="" alt="preview">

        <button id="predictBtn" onclick="predict()" disabled>Predict</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            Analyzing image...
        </div>

        <div class="results" id="results">
            <div class="result-item manufacturer">
                <div class="level">Manufacturer</div>
                <div class="label" id="manuf-label"></div>
                <div class="confidence" id="manuf-conf"></div>
                <div class="bar-bg"><div class="bar-fill" id="manuf-bar" style="width:0%"></div></div>
                <div class="top3" id="manuf-top3"></div>
            </div>
            <div class="result-item family">
                <div class="level">Family</div>
                <div class="label" id="family-label"></div>
                <div class="confidence" id="family-conf"></div>
                <div class="bar-bg"><div class="bar-fill" id="family-bar" style="width:0%"></div></div>
                <div class="top3" id="family-top3"></div>
            </div>
            <div class="result-item variant">
                <div class="level">Variant</div>
                <div class="label" id="variant-label"></div>
                <div class="confidence" id="variant-conf"></div>
                <div class="bar-bg"><div class="bar-fill" id="variant-bar" style="width:0%"></div></div>
                <div class="top3" id="variant-top3"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        // Show image preview when user selects a file
        function previewImage(event) {
            selectedFile = event.target.files[0];
            if (!selectedFile) return;
            const reader = new FileReader();
            reader.onload = e => {
                const preview = document.getElementById('preview');
                preview.src = e.target.result;
                preview.style.display = 'block';
                document.getElementById('predictBtn').disabled = false;
                document.getElementById('results').style.display = 'none';
            };
            reader.readAsDataURL(selectedFile);
        }

        // Send image to Flask /predict endpoint and display results
        async function predict() {
            if (!selectedFile) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('predictBtn').disabled = true;

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();

                // Fill in results for each classification level
                ['manufacturer', 'family', 'variant'].forEach(level => {
                    const r = data[level];
                    const prefix = level === 'manufacturer' ? 'manuf' : level;
                    document.getElementById(`${prefix}-label`).textContent = r.label;
                    document.getElementById(`${prefix}-conf`).textContent = `Confidence: ${r.confidence.toFixed(1)}%`;
                    document.getElementById(`${prefix}-bar`).style.width = r.confidence + '%';
                    const top3div = document.getElementById(`${prefix}-top3`);
                    top3div.innerHTML = r.top3.map(t =>
                        `<div class="top3-item"><span>${t.label}</span><span>${t.confidence.toFixed(1)}%</span></div>`
                    ).join('');
                });

                document.getElementById('results').style.display = 'block';
            } catch(e) {
                alert('Error: ' + e.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
'''
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_name = file.filename
    img = Image.open(io.BytesIO(file.read())).convert('RGB')

    temp_path = BASE_PATH + '/temp_predict.jpg'
    img.save(temp_path)

    results, _ = predict_image(temp_path, feature_extractor, vit, device)
    save_prediction(image_name, results)

    response = {}
    for level in ['manufacturer', 'family', 'variant']:
        response[level] = results[level]

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, port=5000)