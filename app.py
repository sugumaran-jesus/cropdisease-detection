import os
import gdown
import numpy as np
import tensorflow as tf
import uuid                          # ✅ add this
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename   # ✅ add this
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Speed up CPU prediction
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download model from Google Drive if not exists
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(
        "https://drive.google.com/uc?id=1JqrQB23cZVjXMnxEQGntzLhHzEGxXWL0",
        MODEL_PATH, quiet=False
    )

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class names
class_names = sorted([
    'Pepper__bell__Bacterial_spot',
    'Pepper__bell__healthy',
    'Potato__Early_blight',
    'Potato__healthy',
    'Potato__Late_blight',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
])

@app.route('/')
def index():
    return render_template('index.html')

# ✅ REPLACE YOUR OLD PREDICT ROUTE WITH THIS
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', result="❌ No file uploaded")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', result="❌ No file selected")

        # Generate safe unique filename
        ext = file.filename.rsplit('.', 1)[-1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        confidence = round(float(np.max(prediction)) * 100, 2)
        result = class_names[np.argmax(prediction)]
        result_text = result.replace('_', ' ').replace('  ', ' ')

        return render_template('index.html',
                               result=result_text,
                               confidence=confidence,
                               img_filename=filename)

    except Exception as e:
        return render_template('index.html', result=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)