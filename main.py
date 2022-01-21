# importing the modules
import os

import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cv2
import keras

# loading model and data
model = keras.models.load_model('Model')
images = np.load("images.npy")
classNo = np.load("classes.npy")


# preprocessing
def preProcessing(image):
    img = cv2.equalizeHist(image)
    img = img / 255
    return img


# getting the prediction
def get_prediction(roi):
    img = np.asarray(roi)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    predict_x = model.predict(img)
    classes_x = np.argmax(predict_x, axis=1)
    probVal = np.amax(predict_x)

    return classes_x, probVal


main = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
image = cv2.imread("asl_alphabet_test/A_test.jpg")
ind, acc = get_prediction(image)
image = cv2.resize(image, (640,480))
cv2.putText(image,"In ASL:"+main[ind[0]], (50,50), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0), 5)

cv2.imshow("Image", image)
cv2.waitKey(0)

