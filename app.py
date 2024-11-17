from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Set up path for uploaded images and model file
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model_path = 'models/CancerdetectionModel.keras'

# Load the pre-trained model
model = load_model(model_path)

# Function to preprocess and predict using the model
def predict_image(filepath):
    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_array)

    # Determine result based on prediction
    if prediction[0] > 0.5:
        return "Cancerous"
    else:
        return "Non-cancerous"

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run the prediction function
        result = predict_image(filepath)

        # Pass result and image path to the template
        return render_template('index.html', result=result, image_path='uploads/' + file.filename)
    
    return render_template('index.html')

if __name__ == "__main__":
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
