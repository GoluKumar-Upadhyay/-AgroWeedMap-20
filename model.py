import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__)


app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  


for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)


MODEL_PATH = "best_20class_farming_model.keras"
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def apply_clahe_rgb(image):
    image = np.array(image).astype("uint8")
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image.astype("float32")


class_map = {
    0: ("Bermuda grass", "Cynodon dactylon"),
    1: ("Boerhavia erecta", "Boerhavia erecta"),
    2: ("Broadleaf plantain", "Plantago major"),
    3: ("Cannabis sativa", "Cannabis sativa"),
    4: ("Chenopodium album", "Bathua / Lamb's quarters"),
    5: ("Common cocklebur", "Xanthium strumarium"),
    6: ("Creeping woodsorrel", "Oxalis corniculata"),
    7: ("Dhaniya (Coriander)", "Coriandrum sativum"),
    8: ("Goosegrass", "Eleusine indica"),
    9: ("Launaea", "Launaea sarmentosa"),
    10: ("Maize", "Zea mays"),
    11: ("Mustard", "Brassica juncea"),
    12: ("Parthenium", "Parthenium hysterophorus"),
    13: ("Pigweed Amaranthus", "Amaranthus spp."),
    14: ("Potato", "Solanum tuberosum"),
    15: ("Prostrate spurge", "Euphorbia prostrata"),
    16: ("Purple nutsedge", "Cyperus rotundus"),
    17: ("Sesbania", "Sesbania bispinosa"),
    18: ("Sowthistle", "Sonchus oleraceus"),
    19: ("Tomato", "Solanum lycopersicum")
}


def softmax_temperature(probs, T=3.0):
    logits = np.log(probs + 1e-9)
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits)

def prediction_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        return unique_filename, filepath
    return None, None


def predict_image(image_path, conf_threshold=75, entropy_threshold=1.2):
    """Predict plant from image and return results"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((224, 224))
        
        img_array = apply_clahe_rgb(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        raw_preds = model.predict(img_array, verbose=0)[0]
        preds = softmax_temperature(raw_preds, T=3.0)
        
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx] * 100
        entropy = prediction_entropy(preds)
        
        
        top_k = 3
        top_indices = np.argsort(preds)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            label_text, sci_text = class_map.get(idx, ("Unknown", "Unknown"))
            top_predictions.append({
                "name": label_text,
                "scientific_name": sci_text,
                "confidence": float(preds[idx] * 100),
                "class_id": int(idx)
            })
        
        
        if confidence < conf_threshold or entropy > entropy_threshold:
            final_label = "Unknown Plant ❓"
            final_scientific = "Not in trained categories"
            is_unknown = True
        else:
            final_label, final_scientific = class_map[pred_idx]
            is_unknown = False
        
        return {
            "success": True,
            "prediction": {
                "label": final_label,
                "scientific_name": final_scientific,
                "confidence": float(confidence),
                "entropy": float(entropy),
                "class_id": int(pred_idx),
                "is_unknown": is_unknown
            },
            "top_predictions": top_predictions,
            "image_size": img.size
        }
    
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    """Render the main page"""
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    
    filename, filepath = save_uploaded_file(file)
    if not filename:
        return jsonify({'error': 'Invalid file type'})
    
    
    result = predict_image(filepath)
    
    
    if 'error' in result:
        return jsonify({'error': result['error']})
    
    
    result['filename'] = filename
    result['file_url'] = f'/uploads/{filename}'
    
    return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(debug=True, host='0.0.0.0', port=port)