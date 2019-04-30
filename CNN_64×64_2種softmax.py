#ML関連import
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
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

image_list = []		#画像情報を入れる配列
label_list = []		#画像のラベルを入れる配列
val_image_list = []     #validation用の配列
val_label_list = []     #同上

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
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            image_preprocessed = cv2.GaussianBlur(image_gray, (5, 5), 0)	#画像のぼかし
            #閾値処理
            _, image_binary = cv2.threshold(image_preprocessed, 50, 255, cv2.THRESH_BINARY)
            
            #マスク処理
            img_masked = cv2.bitwise_and(image, image, mask=image_binary) 
            #cv2.imshow('bina', img_masked)  
            """
            black = [0,0,0]   #blackに黒色の情報
            white = [255, 255, 255]  #whiteに白色の情報
            img_masked[np.where((img_masked == black).all(axis=2))] = white   #img_maskedのblack(黒色)と一致するピクセルをwhite(白色)に変換
            #img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
            image = img_masked	#マスク処理した画像でimageを上書き
            cv2.imshow("gaisetu", image)
            cv2.waitKey(0)
            """
            image = cv2.resize(img_masked, (64, 64))		#画像の縮小
            #cv2.imshow("gaisetu", image)
            #cv2.waitKey(0)
            #画像処理終了-------------------------------------------------------------------------------------------------------------------------
            
            #画像をnumpy配列に
            image = np.array(image)
            print(image.shape)
            print(filepath)
            image_list.append(image / 255.)


#validation_dataの読み込み
for dir in os.listdir("/home/pi/data/val/"):
    if dir == ".DS_Store":
        continue

    dir1 = "/home/pi/data/val/" + dir 
    label = 0

    if dir == "ari":    # ariはラベル0
        label = 0
    elif dir == "nashi": # nashiはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加
            val_label_list.append(label)
            filepath = dir1 + "/" + file
            #ここからopencvによる画像処理-----------------------------------------------------------------------------------------------------
            val_image = cv2.imread(filepath)
            val_image_gray = cv2.cvtColor(val_image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            val_image_preprocessed = cv2.GaussianBlur(val_image_gray, (5, 5), 0)	#画像のぼかし
            #閾値処理
            _, val_image_binary = cv2.threshold(val_image_preprocessed, 50, 255, cv2.THRESH_BINARY)

            #マスク処理
            val_img_masked = cv2.bitwise_and(val_image, val_image, mask=val_image_binary)  
            """ 
            black = [0,0,0]   #blackに黒色の情報
            white = [255, 255, 255]  #whiteに白色の情報
            val_img_masked[np.where((val_img_masked == black).all(axis=2))] = white   #img_maskedのblack(黒色)と一致するピクセルをwhite(白色)に変換
            val_img_masked = cv2.cvtColor(val_img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
            val_image = val_img_masked	#マスク処理した画像でimageを上書き
            """
            val_image = cv2.resize(val_img_masked, (64, 64))		#画像の縮小
            #cv2.imshow("gaisetu", val_image)
            #cv2.waitKey(0)
            #画像処理終了-------------------------------------------------------------------------------------------------------------------------
            
            #画像をnumpy配列に
            val_image = np.array(val_image)
            print(val_image.shape)
            print(filepath)
            val_image_list.append(val_image / 255.)




image_list = np.array(image_list)
label_list = to_categorical(label_list)
val_image_list = np.array(val_image_list)
val_label_list = to_categorical(val_label_list)

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(64,64,3)))
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

adam = Adam(lr=1e-3)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',  patience=3, verbose=0, mode='auto')

history = model.fit(image_list, label_list, epochs=20, batch_size=50, validation_data=(val_image_list, val_label_list), callbacks=[early_stopping])

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

for dir in os.listdir("/home/pi/data/test/"):
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
            #image = image[0:700, 60:760]
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            image_preprocessed = cv2.GaussianBlur(image_gray, (5, 5), 0)	#画像のぼかし
            #閾値処理
            _, image_binary = cv2.threshold(image_preprocessed, 50, 255, cv2.THRESH_BINARY)

            #マスク処理
            img_masked = cv2.bitwise_and(image, image, mask=image_binary)
            """   
            black = [0,0,0]   #blackに黒色の情報
            white = [255, 255, 255]  #whiteに白色の情報
            img_masked[np.where((img_masked == black).all(axis=2))] = white   #img_maskedのblack(黒色)と一致するピクセルをwhite(白色)に変換
            #img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
            """
            image = img_masked	#マスク処理した画像でimageを上書き
            image = cv2.resize(image, (64, 64))		#画像の縮小
            #---------------------------------------------------------------------------------
            result = model.predict_classes(np.array([image / 255.]))
            result2 = model.predict(np.array([image / 255.]))
            print(result2)
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")
