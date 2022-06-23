import cv2
import os, glob
import keras
from keras.models import Sequential
from keras.models import load_model
from flowerLanguage import flowerLanguage
import numpy as np
import matplotlib.pyplot as plt
obs_labels = [ "무궁화",  "백합",  "철쭉", "진달래" ,"벚꽃" ,"연꽃"]


def use_pc():
    new_obs_model = load_model('obs_model.h5')
    img_path = 'D:\machine\mr\capture/'
    img_files = glob.glob(img_path + "/*.jpg")
    for img in img_files:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        predicted_result = new_obs_model.predict_generator(np.array([frame]))
        obs_predicted = predicted_result[0]
        print('꽃 이름 = ', obs_labels[obs_predicted.argmax()])
        flowerLanguage(obs_labels[obs_predicted.argmax()])
        plt. imshow(frame)
        tmp = " Prediction:" + obs_labels[obs_predicted.argmax()]
        plt.title(tmp)  
        plt.show()
        

new_obs_model = load_model('obs_model.h5')
use_pc()