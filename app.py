from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
import keras.utils as image
# from keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# Ganti dengan jalur menuju model CNN Anda
model = tf.keras.models.load_model('./model/model_cnn.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    kelas = ''
    file = request.files['file']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    # Ganti dengan ukuran gambar yang sesuai dengan model Anda
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalisasi gambar
    prediction = model.predict(img)
    # Lakukan sesuatu dengan hasil prediksi, misalnya mengambil label kelas teratas
    predicted_class = np.argmax(prediction)
    # ...
    if predicted_class == 0:
        kelas = "Diabetic Retinopathy"
    elif predicted_class == 1:
        kelas = "Hypertensive Retinopathy"
    elif predicted_class == 2:
        kelas = "Normal"

    return render_template('index.html', prediction=kelas, file_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
