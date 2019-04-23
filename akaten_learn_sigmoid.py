#ML関連import
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import h5py

#画像関連import
import cv2
import matplotlib.pyplot as plt

image_list = []		#画像情報を入れる配列
label_list = []		#画像のラベルを入れる配列


#指定したフォルダの画像をすべて読み込む➡opencvで検出、輪郭描写、背景白潰しを行う。　
for dir in os.listdir("/home/pi/data/train/"):
    if dir == ".DS_Store":
        continue

    dir1 = "/home/pi/data/train/" + dir 
    label = 0

    if dir == "ari":    # ariはラベル0
        label = 0
    elif dir == "nashi": # nashiはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加
            label_list.append(label)
            filepath = dir1 + "/" + file
            #ここからopencvによる画像処理-----------------------------------------------------------------------------------------------------
            image = cv2.imread(filepath)
            image = image[0:670, 0:768] #画像の端を少しカット
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            image_preprocessed = cv2.GaussianBlur(image_gray, (5, 5), 0)	#画像のぼかし
            #閾値処理
            _, image_binary = cv2.threshold(image_preprocessed, 90, 255, cv2.THRESH_BINARY)

            #マスク処理
            img_masked = cv2.bitwise_and(image, image, mask=image_binary)   
            black = [0,0,0]   #blackに黒色の情報
            white = [255, 255, 255]  #whiteに白色の情報
            img_masked[np.where((img_masked == black).all(axis=2))] = white   #img_maskedのblack(黒色)と一致するピクセルをwhite(白色)に変換
            img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
            image = img_masked	#マスク処理した画像でimageを上書き
            image = cv2.resize(image, (25, 25))		#画像の縮小
            #画像処理終了-------------------------------------------------------------------------------------------------------------------------
            
            #画像をnumpy配列に
            image = np.array(image)
            print(image.shape)
            image = image.reshape(-1,)
            print(image.shape)
            print(filepath)
            image_list.append(image / 255.)

image_list = np.array(image_list)
label_list = to_categorical(label_list)

model = Sequential()

model.add(Conv2D(32, (3,3), padding="same", input_shape=(1875,)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3),  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(image_list, label_list, epochs=15, batch_size=100, validation_split=0.1)

model.save("akaten_learn_model.h5")


#accuracyのグラフ（関数変更予定）
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#lossのグラフ　（関数変更予定）
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

total = 0.
ok_count = 0.
image_list = []
label_list = []
for dir in os.listdir("/home/pi/data/train/"):
    if dir == ".DS_Store":
        continue

    dir1 = "/home/pi/data/test/" + dir 
    label = 0

    if dir == "ari":    # ariはラベル0
        label = 0
    elif dir == "nashi": # nashiはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加
            label_list.append(label)
            filepath = dir1 + "/" + file
            #ここからopencvによる画像処理-----------------------------------------------------------------------------------------------------
            image = cv2.imread(filepath)
            image = image[0:670, 0:768] #画像の端を少しカット
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            image_preprocessed = cv2.GaussianBlur(image_gray, (5, 5), 0)	#画像のぼかし
            #閾値処理
            _, image_binary = cv2.threshold(image_preprocessed, 90, 255, cv2.THRESH_BINARY)

            #マスク処理
            img_masked = cv2.bitwise_and(image, image, mask=image_binary)   
            black = [0,0,0]   #blackに黒色の情報
            white = [255, 255, 255]  #whiteに白色の情報
            img_masked[np.where((img_masked == black).all(axis=2))] = white   #img_maskedのblack(黒色)と一致するピクセルをwhite(白色)に変換
            img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
            image = img_masked	#マスク処理した画像でimageを上書き
            image = cv2.resize(image, (32, 32))		#画像の縮小
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")
