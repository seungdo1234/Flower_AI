
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil
import downloader
from torchvision import datasets, models, transforms
import numpy as np
import time

directory_list = [
    'D:\machine\mr/train/',
    'D:\machine\mr/test/',
]


for directory in directory_list:
    if not os.path.isdir(directory):
        os.makedirs(directory)

def dataset_split(query, train_cnt):

    for directory in directory_list:
        if not os.path.isdir(directory + '/' + query):
            os.makedirs(directory + '/' + query)
    for file_name in os.listdir(query):
        print(f'[Train Dataset] {file_name}')
        shutil.move(query + '/' + file_name, '\machine\mr/train/' + query + '/' + file_name)
    shutil.rmtree(query)

query = '개나리'
downloader.download(query, limit=30,  output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
dataset_split(query, 30)

    

