import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter
from keras.models import Sequential
from tkinter import *
import tensorflow as tf
from tensorflow import keras

def guess():
    vdo = cv.VideoCapture(0)
    userImg = vdo.read()
    vdo.release()

    userImg = cv.cvtColor(userImg, cv.COLOR_BGR2GRAY)
    userImg = cv.resize(userImg, (48, 36), interpolation = cv.INTER_CUBIC)

    np_img = np.asarray(userImg)
    np_img = np.delete(np_img, 2, 2)
    np_img = np.delete(np_img, 1, 2)
    np_img = np.reshape(np_img, (36, 48, 1))

    model = keras.models.load_model('D:\\Python\\hand maching\\test-model-epoch5.h5')
    probability_model = Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(np_img)
    np.argmax(predictions[0])
    if predictions[0]==0:
        guess = Label(frame, text = "Palm(Vertical)")
    elif predictions[0]==1:
        guess = Label(frame, text = "Palm(Horizontal)")
    elif predictions[0]==2:
        guess = Label(frame, text = "1 finger")
    elif predictions[0]==3:
        guess = Label(frame, text = "Fist")
    elif predictions[0]==4:
        guess = Label(frame, text = "Thumbs Up")
    guess.pack()


root = Tk()
root.geometry("640x480")
frame = Frame(root)
frame.pack()

label = Label(frame, text = "Click to Guess your Hand Gesture!", command = guess())
label.pack()

testButton = Button(frame, text = "Guess")
testButton.pack(padx = 3, pady = 3)

root.title("Test")
root.mainloop()
