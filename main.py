from flask import Flask, render_template, request, send_from_directory
from keras.api.preprocessing.image import load_img, img_to_array
from keras.api.models import load_model
import tensorflow as tf
import numpy as np
import os
from PIL import Image
#Create App
app = Flask(__name__)

#load the trained model
model = load_model('models\model.h5')

#Class labels
class_labels = ['PITUITARY', 'GLIOMA', 'NO TUMOR', 'MENINGIOMA']

# Define the uploads folder
upload_folder = "./uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['uploads'] = upload_folder

#Helper function to predict tumor type
def predict_tumor(img_path):
    image_size = 255
    img = load_img(img_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]

    if class_labels[predicted_class_index] == 'NO TUMOR':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

#COMMUNICATION BETWEEN FRONTEND AND BACKEND#
# Routes
@app.route("/", methods=['GET', 'POST'])
def index():
    #Handle file upload
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # save the file
            file_location = os.path.join('uploads', file.filename)
            file.save(file_location)

            #Predict the tumor
            result , confidence_score = predict_tumor(file_location)

            #return results along with image path for display
            return render_template('index.html', result=result, confidence_score=f'{confidence_score*100:.2f}%', file_location=f'/uploads/{file.filename}')

    return render_template('index.html', result="NULL", confidence_score='0')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['uploads'], filename)

if __name__ == '__main__':
    app.run(debug=True)

