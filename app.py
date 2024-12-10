from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('models/best_model.keras')
class_labels = ['Normal', 'Cataract', 'Glaucoma', 'Diabetic Retinopathy']

app = Flask(__name__)

if not os.path.exists('images'):
    os.makedirs('images')

@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')

@app.route('/process', methods=['GET'])
def process():
    return render_template('process.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    imageFile = request.files['imageFile']
    image_path = os.path.join('images', imageFile.filename)
    imageFile.save(image_path)

    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    predicted_confidence = predictions[0][predicted_class_index] * 100

    print(f"Image Path: {image_path}")
    print(f"Predicted Class: {predicted_class} with {predicted_confidence:.2f}% confidence")

    return render_template(
        'main.html', 
        prediction=predicted_class, 
        confidence=f"{predicted_confidence:.2f}"
    )

if __name__ == '__main__':
    app.run(port=3000, debug=True)