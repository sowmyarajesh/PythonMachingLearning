import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter
from keras.models import Sequential
import tkinter as tk
import tensorflow as tf
from tensorflow import keras

def guess():
    vdo = cv.VideoCapture(0+cv.CAP_DSHOW)
    retval,userImg = vdo.read()
    vdo.release()
    # Destroy all the windows
    cv.destroyAllWindows()
    if retval != True:
        print("Can't read frame")
        return
    userImg = cv.cvtColor(userImg, cv.COLOR_BGR2GRAY)
    userImg = cv.resize(userImg, (48, 36), interpolation = cv.INTER_CUBIC)
    cv.imshow('frame',userImg)

    np_img = np.asarray(userImg)
    
    # should you delete row / column? the 3rd argument in this delete function can only be 0=row, 1-=col
    # you mentioned here 2 which is  out of bounds
    np_img = np.delete(np_img, 2, 2)
    np_img = np.delete(np_img, 1, 2)
    np_img = np.reshape(np_img, (36, 48, 1))

    model = keras.models.load_model('D:\\Python\\hand maching\\test-model-epoch5.h5')
    probability_model = Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(np_img)
    np.argmax(predictions[0])
    if predictions[0]==0:
        guess = tk.Label(frame, text = "Palm(Vertical)")
    elif predictions[0]==1:
        guess = tk.Label(frame, text = "Palm(Horizontal)")
    elif predictions[0]==2:
        guess = tk.Label(frame, text = "1 finger")
    elif predictions[0]==3:
        guess = tk.Label(frame, text = "Fist")
    elif predictions[0]==4:
        guess = tk.Label(frame, text = "Thumbs Up")
    guess.pack()

if __name__=='__main__':
    root = tk.Tk()
    root.geometry("640x480")
    frame = tk.Frame(root)
    frame.pack()

    label = tk.Label(frame, text = "Click to Guess your Hand Gesture!", command = guess())
    label.pack()

    testButton = tk.Button(frame, text = "Guess")
    testButton.pack(padx = 3, pady = 3)

    root.title("Test")
    root.mainloop()

