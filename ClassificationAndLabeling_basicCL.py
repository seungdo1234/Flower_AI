import cv2
import os, glob
import numpy
import matplotlib.pyplot as plt
import downloader
import shutil

image_size = (128,128)
image_folder = 6;
path = os.path.dirname(os.path.abspath(__file__))

obs_Xdata = [] 
obs_Ylabel = []
obs_obs = []

for i in range(0,image_folder):
    path_obs =path + '/test' + str(i)
    obs_img_files = glob.glob(path_obs + "/*.jpg")
    for j in obs_img_files:
        obs_img = cv2.imread(j)
        obs_img = cv2.resize(obs_img, image_size)
        obs_Xdata.append(obs_img)
        obs_Ylabel.append(i)


    
