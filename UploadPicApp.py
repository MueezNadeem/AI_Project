from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('trained_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32')
    img /= 255
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    
    return str(predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
