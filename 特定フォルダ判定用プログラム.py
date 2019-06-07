#ML関連import
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import os
import h5py

#画像関連import
import cv2
import matplotlib.pyplot as plt

import sys
from picamera import PiCamera
from time import sleep

model = load_model('6_6_3syu_1080_96x96_model.h5')
kanri = 0
count = 1


total = 0.
ok_count = 0.
image_list = []
label_list = []

for dir in os.listdir("/home/pi/hantei/hantei/"):
    if dir == ".DS_Store":
        continue

    dir1 = "/home/pi/hantei/hantei/" + dir 
    label = 0

    if dir == "ari":    # ariはラベル0
        label = 0
    elif dir == "nasi": # nashiはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加
            label_list.append(label)
            filepath = dir1 + "/" + file
            #ここからopencvによる画像処理-----------------------------------------------------------------------------------------------------
            image = cv2.imread(filepath)
            #image = image[0:700, 60:760]
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            image_preprocessed = cv2.GaussianBlur(image_gray, (5, 5), 0)	#画像のぼかし
            #閾値処理
            _, image_binary = cv2.threshold(image_preprocessed, 40, 255, cv2.THRESH_BINARY)

            #マスク処理
            img_masked = cv2.bitwise_and(image, image, mask=image_binary)
            """   
            black = [0,0,0]   #blackに黒色の情報
            white = [255, 255, 255]  #whiteに白色の情報
            img_masked[np.where((img_masked == black).all(axis=2))] = white   #img_maskedのblack(黒色)と一致するピクセルをwhite(白色)に変換
            #img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
            """
            image = img_masked	#マスク処理した画像でimageを上書き
            image = cv2.resize(image, (96, 96))		#画像の縮小
            #cv2.imshow('resize', image)
            #cv2.waitKey(0)
            #---------------------------------------------------------------------------------
            result = model.predict_classes(np.array([image / 255.]))
            result2 = model.predict(np.array([image / 255.]))
            print(filepath)
            print(result2)
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")
