import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter,ImageTk
from keras.models import Sequential
from tkinter import *
import tensorflow as tf
from tensorflow import keras

guessLabel = None

def guess():
    if guessLabel is None:
        guessLabel = Label(frame, text = "")
    # guessLabel.pack()
    
    vdo = cv.VideoCapture(0+cv.CAP_DSHOW)
    retval,userImg = vdo.read()
    vdo.release()
    # Destroy all the windows
    cv.destroyAllWindows()
    if retval != True:
        print("Can't read frame")
        guessLabel = Label(frame, text = "cant read")
        guessLabel.pack()
        return 5
    userImg = cv.cvtColor(userImg, cv.COLOR_BGR2GRAY)    
    # blue,green,red = cv.split(userImg)
    # img = cv.merge((red,green,blue))
    # im = Image.fromarray(img)
    # imgtk = ImageTk.PhotoImage(image=im)

    cv.imshow('frame',userImg)
    userImg = cv.resize(userImg, (48, 36), interpolation = cv.INTER_CUBIC)

    np_img = np.asarray(userImg)
    # print(np_img)
    # np_img = np.delete(np_img, 2, 2)
    # np_img = np.delete(np_img, 1, 2)
    np_img = np.reshape(np_img, (1, 36, 48, 1))

    model = keras.models.load_model(os.getcwd() + '\\test-model-epoch5.h5')
    probability_model = Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(np_img)
    print(predictions)
    np.argmax(predictions)
    print("argmax: ",np.argmax(predictions))
    # guessLabel.destroy()
    if np.argmax(predictions[0])==0:
        # guessLabel = Label(frame, text = "Palm(Vertical)")
        guessLabel.config(text="Palm(Vertical)")
    elif np.argmax(predictions[0])==1:
        # guessLabel = Label(frame, text = "Palm(Horizontal)")
        guessLabel.config(text="Palm(Horizontal)")
    elif np.argmax(predictions[0])==2:
        # guessLabel = Label(frame, text = "1 finger")
        guessLabel.config(text="1 finger")
    elif np.argmax(predictions[0])==3:
        # guessLabel = Label(frame, text = "Fist")
        guessLabel.config(text="Fist")
    elif np.argmax(predictions[0])==4:
        # guessLabel = Label(frame, text = "Thumbs Up")
        guessLabel.config(text="Thumbs Up")
    # guessLabel.grid(column=1,row=1)
    guessLabel.pack()


root = Tk()
root.geometry("640x480")

# root.columnconfigure(0, minsize=250)
# root.rowconfigure([0, 1], minsize=100)
frame = Frame(root)
frame.pack()
label = Label(frame, text = "Click to Guess your Hand Gesture!")
label.pack()

testButton = Button(frame, text = "Guess", command = guess)
# testButton.grid(row=0, column=1, sticky="w")
testButton.pack(padx = 3, pady = 3)

root.title("Guess Gesture")
root.mainloop()
