from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
import imutils
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

class DisplayTumor:
    def __init__(self):
        self.curImg = None
        self.Img = None
        self.thresh = None
        self.kernel = None
        self.ret = None

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        return self.curImg

    def removeNoise(self):
        self.kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]
        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage

def predictTumor(image):
    model = load_model('brain_tumor_detector.h5')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    image = image / 255.
    image = image.reshape((1, 240, 240, 3))
    res = model.predict(image)
    return res[0][0]

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    option = request.form.get('option')
    img = Image.open(file).convert('RGB')
    img_array = np.array(img)

    dt = DisplayTumor()
    dt.readImage(img_array)

    if option == 'detect':
        res = predictTumor(img_array)
        message = 'Tumor Detected' if res > 0.5 else 'No Tumor'
        img_array = dt.getImage()
    elif option == 'visualize':
        dt.removeNoise()
        dt.displayTumor()
        message = 'Tumor Region Visualized'
        img_array = dt.getImage()
    else:
        return jsonify({'error': 'Invalid option'}), 400

    _, buffer = cv.imencode('.jpg', img_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_base64, 'message': message})

if __name__ == '__main__':
    app.run(debug=True, port=8087)
