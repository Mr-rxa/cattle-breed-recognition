from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging

app = Flask(__name__)
CORS(app)

# Load model and labels
model = tf.keras.models.load_model('models/best_cattle_breed_model.h5')
with open('models/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Hindi translations
hindi_names = {
    "Alambadi": "अलंबाड़ी",
    "Amritmahal": "अमृतमहल",
    "Ayrshire": "एयरशायर",
    "Banni": "बन्नी",
    "Bargur": "बारगुर",
    "Bhadawari": "भदावरी",
    "Brown_Swiss": "ब्राउन स्विस",
    "Dangi": "दांगी",
    "Deoni": "डियोनी",
    "Gir": "गिर",
    "Guernsey": "ग्वर्नसी",
    "Hallikar": "हल्लीकर",
    "Hariana": "हरियाना",
    "Holstein_Friesian": "होल्स्टीन फ्रिसियन",
    "Jaffrabadi": "जाफराबादी",
    "Jersey": "जर्सी",
    "Kangayam": "कांगयम",
    "Kankrej": "कांकेज",
    "Kasargod": "कसारगोड़",
    "Kenkatha": "केनकथा",
    "Kherigarh": "खेड़िगढ़",
    "Khillari": "खिल्लारी",
    "Krishna_Valley": "कृष्णा वैली",
    "Malnad_gidda": "मलनाड गिद्दा",
    "Mehsana": "मेहसाणा",
    "Murrah": "मुर्हा",
    "Nagori": "नागोरी",
    "Nagpuri": "नागपुरी",
    "Nili_Ravi": "नीली रवि",
    "Nimari": "निमाड़ी",
    "Ongole": "ओंगोल",
    "Pulikulam": "पुलीकुलम",
    "Rathi": "राठी",
    "Red_Dane": "रेड डेन",
    "Red_Sindhi": "रेड सिंधी",
    "Sahiwal": "साहीवाल",
    "Surti": "सुरती",
    "Tharparkar": "थारपारकर",
    "Toda": "तोड़ा",
    "Umblachery": "उम्ब्लाचेरी",
    "Vechur": "वचूर"
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    return arr.reshape((1, 224, 224, 3))

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>🐄 Bharat Pashudhan – Breed Recognition</title>
  <style>
    body { font-family: "Segoe UI", sans-serif; background: #f0f4f8; color: #1e293b;
           text-align: center; max-width: 600px; margin: 0 auto; padding: 40px 20px 60px; }
    h1 { color: #2563eb; }
    p.subtitle { color:#475569; font-size:1.1rem; margin-bottom:30px; }

    /* Upload Box */
    .upload-box {
      background: #fff;
      border: 3px dashed #94a3b8;
      border-radius: 16px;
      padding: 50px 30px;
      margin-bottom: 20px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 220px;
      transition: border-color 0.3s, background 0.3s;
    }
    .upload-box:hover { background: #f9fafb; }
    .upload-box.dragover {
      border-color: #2563eb !important;
      background: #eff6ff !important;
    }
    input[type=file] { display: none; }
    .upload-text { font-size: 1.2rem; color: #475569; text-align:center; }
    .preview img {
      max-width: 250px;
      border-radius: 16px;
      box-shadow: 0 6px 16px rgba(37,99,235,0.2);
      margin-top: 20px;
    }

    /* Buttons */
    button {
      background: #2563eb; color: #fff; padding: 14px 24px; border: none; border-radius: 12px;
      font-size: 1rem; margin: 10px; cursor: pointer; transition: all 0.3s;
      box-shadow: 0 4px 12px rgba(37,99,235,0.4);
      font-weight:600;
    }
    button:disabled { background: #94a3b8; cursor: not-allowed; box-shadow: none; }
    button:hover { background: #1e40af; }

    /* Results */
    .result { opacity: 0; transform: translateY(20px); transition: all 0.6s ease;
              background: #fff; padding: 20px; border-radius: 14px; margin-top: 25px;
              box-shadow: 0 6px 18px rgba(0,0,0,0.1); text-align:left; }
    .result.show { opacity: 1; transform: translateY(0); }
    .breed { display:flex; justify-content:space-between; font-size:1.2rem; font-weight:700; color:#16a34a; }
    .hin { font-style: italic; color:#2563eb; }
    .confidence-bar { background:#e2e8f0; border-radius:12px; overflow:hidden; height:14px; margin:6px 0; }
    .confidence-fill { background:linear-gradient(to right,#16a34a,#22c55e); height:100%; transition:width 0.8s ease; }
  </style>
</head>
<body>
  <h1>🐄 Bharat Pashudhan – Breed Recognition<br/>भारत पशुधन – नस्ल पहचान</h1>
  <p class="subtitle">Upload, Drag & Drop, or Take a Photo<br/>तस्वीर अपलोड करें, खींचें या फोटो लें</p>

  <!-- Upload Box -->
  <label id="drop-area" class="upload-box" for="fileInput">
    <div class="upload-text">
      📂 Click or Drag & Drop Here<br/>क्लिक करें या यहाँ खींचें
    </div>
    <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
    <div class="preview" id="preview"></div>
  </label>

  <button type="button" onclick="openCamera()">📷 Take Photo (फोटो लें)</button>
  <button id="predictBtn" disabled>🔍 Predict Breed (नस्ल पहचानें)</button>

  <!-- Live Camera Section -->
  <div id="cameraSection" style="display:none; text-align:center; margin-top:20px;">
    <video id="video" autoplay playsinline style="max-width:100%; border-radius:12px;"></video><br/>
    <button type="button" onclick="capturePhoto()">📸 Capture (कैप्चर करें)</button>
    <canvas id="canvas" style="display:none;"></canvas>
  </div>

  <!-- Result -->
  <div class="result"></div>

<script>
const fileInput = document.getElementById('fileInput');
const dropArea = document.getElementById('drop-area');
const predictBtn = document.getElementById('predictBtn');
const preview = document.getElementById('preview');
const resultDiv = document.querySelector('.result');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const cameraSection = document.getElementById('cameraSection');
let stream = null;

function previewImage(evt) {
  const file = evt.target.files[0];
  if (!file) { preview.innerHTML=''; predictBtn.disabled=true; return; }
  const reader = new FileReader();
  reader.onload = e => { preview.innerHTML = `<img src="${e.target.result}"/>`; };
  reader.readAsDataURL(file);
  predictBtn.disabled=false;
}

;['dragover','dragleave','drop'].forEach(evtName => {
  dropArea.addEventListener(evtName, e => {
    e.preventDefault();
    if(evtName==='dragover') dropArea.classList.add('dragover');
    else dropArea.classList.remove('dragover');
    if(evtName==='drop' && e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      previewImage({ target: { files: e.dataTransfer.files } });
    }
  });
});

function openCamera() {
  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(s => { stream = s; video.srcObject = stream; cameraSection.style.display="block"; })
    .catch(err => { alert("Camera not available: " + err); });
}

function capturePhoto() {
  const context = canvas.getContext('2d');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  stream.getTracks().forEach(track => track.stop());
  cameraSection.style.display="none";
  const dataUrl = canvas.toDataURL("image/png");
  preview.innerHTML = `<img src="${dataUrl}"/>`;
  fetch(dataUrl).then(res=>res.blob()).then(blob=>{
    const file = new File([blob], "capture.png", {type:"image/png"});
    const dt = new DataTransfer(); dt.items.add(file);
    fileInput.files = dt.files; predictBtn.disabled=false;
  });
}

predictBtn.addEventListener('click', () => {
  if (!fileInput.files.length) return;
  let fd = new FormData(); fd.append('file', fileInput.files[0]);
  predictBtn.disabled=true; predictBtn.textContent="⌛ Processing...";
  resultDiv.classList.remove('show'); resultDiv.innerHTML="";
  fetch('/predict', { method:'POST', body:fd })
    .then(r=>r.json()).then(data=>{
      predictBtn.disabled=false; predictBtn.textContent="🔍 Predict Breed (नस्ल पहचानें)";
      if (data.error) { resultDiv.innerHTML=`<div style='color:red;'>⚠️ ${data.error}</div>`; resultDiv.classList.add('show'); return; }
      let html="<h2>Top Predictions (शीर्ष अनुमान)</h2>";
      data.predictions.forEach((p,i)=>{ html+=`
        <div class="breed"><span>${p.label}</span><span class="hin">${p.hindi}</span></div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${p.confidence*100}%"></div></div>
        <small>Confidence (विश्वास): ${(p.confidence*100).toFixed(1)}%</small>
        ${i<data.predictions.length-1?"<hr/>":""}`; });
      resultDiv.innerHTML=html; setTimeout(()=>resultDiv.classList.add('show'),50);
    }).catch(()=>{ alert("Error communicating with server."); });
});
</script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, hindi_names=hindi_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        img = preprocess_image(request.files['file'].read())
        preds = model.predict(img)[0]
        top = preds.argsort()[-3:][::-1]
        predictions = [
            {'label': class_names[i], 'hindi': hindi_names.get(class_names[i], "---"), 'confidence': float(preds[i])}
            for i in top
        ]
        return jsonify({'predictions': predictions})
    except Exception as e:
        logging.error("Prediction error", exc_info=True)
        return jsonify({'error': '❌ Failed to process image. कृपया दोबारा कोशिश करें'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
