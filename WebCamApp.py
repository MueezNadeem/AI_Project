import cv2
from flask import Flask, render_template, Response
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('trained_model.h5') 
def predict_digit(image):
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.reshape(image, (1, 28, 28, 1))
    image = image.astype('float32') / 255

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class


def generate_frames():
    video_capture = cv2.VideoCapture(0)  
    while True:
        _, frame = video_capture.read()  
        digit = predict_digit(frame)

        cv2.putText(frame, str(digit), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
