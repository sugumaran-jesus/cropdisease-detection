import os
import gdown
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download model from Google Drive
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=1JqrQB23cZVjXMnxEQGntzLhHzEGxXWL0",
        MODEL_PATH, quiet=False
    )

model = load_model(MODEL_PATH)

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    return render_template('index.html', result=result, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)