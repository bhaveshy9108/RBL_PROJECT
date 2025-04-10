from flask import Flask, render_template, request,send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from PIL import Image


app = Flask(__name__)

# Load the trained model
model_path = "./custom_model_combined.h5"
model = load_model(model_path)

# Define the target image size
target_size = (128, 128)

# Define the class labels
CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Function to preprocess the input image
# def preprocess_image(image_file):
#     img = Image.open(image_file)
#     img = img.resize(target_size)
#     img = np.array(img) / 255.0  # Normalize pixel values
#     img = np.expand_dims(img, axis=0)
#     return img


#Trial Part 1:-
# def preprocess_new_image(image_path, target_size=(128, 128)):
#     img = image.load_img(image_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(128, 128))  # Load image with desired target size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

@app.route('/static/images/<path:filename>')
def static_images(filename):
    try:
        return send_from_directory('static/images', filename)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {e}")
        return "Error serving image", 500

@app.route('/static/videos/<path:filename>')
def static_videos(filename):
    try:
        return send_from_directory('static/videos', filename)
    except Exception as e:
        app.logger.error(f"Error serving video {filename}: {e}")
        return "Error serving video", 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file temporarily
            filename = "temp.jpg"
            filepath = os.path.join(app.root_path, "uploads", filename)
            file.save(filepath)  # Provide the filepath here, not just the directory

            # Preprocess the image
            img = preprocess_image(filepath)

            # Predict
            prediction = model.predict(img)[0]
            predicted_class_index = np.argmax(prediction)
            predicted_class = CLASSES[predicted_class_index]
            confidence = prediction[predicted_class_index]
            # Remove the temporary file
            os.remove(filepath)

            return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
    # Handle GET request or other methods
    return render_template('predict.html')

# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             img = preprocess_image(file)
#             prediction = model.predict(img)[0]
#             predicted_class_index = np.argmax(prediction)
#             predicted_class = CLASSES[predicted_class_index]
#             confidence = prediction[predicted_class_index]
#             return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
#     # Handle GET request or other methods
#     return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
