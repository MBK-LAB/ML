from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image



save_path = 'output'  # 保存ディレクトリのパス
#dir1 = "/home/pi/camera" + dir 

for file in os.listdir("/home/pi/input"):
        if file != ".DS_Store":
            filepath = "/home/pi/input" + "/" + file        #読み込むディレクトリ(フォルダ)を指定
            image = cv2.imread(filepath)           #指定されたファイルを読み込む
            image = image[0:700, 60:760]           #画像のカット(ほしい対象物でない部分をカットしている)　全部必要なら編集して飛ばす
            """
            cv2.imshow("coins", image)          #カットした画像確認用の画像表示
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        #opencvで読み込むとBGRデータなのでRGBに直しておく
            print(image.shape)       #imageの形を表示　（size,size,channel数）で表示
            
            """
            image1 = image1.convert("RGB")            #opencv使わない時はこっち
            image1 = image1.resize((32, 32))
            image = Image.open(filepath)
            """
            
            print(filepath)               #現在のファイルを表示
            image1 = np.array(image)      #imageをnumpyに変更
            print(image1.shape)
            
            image1 = image1[np.newaxis]  　#image1を4次元情報に変更 
            datagen = ImageDataGenerator(rotation_range=90, vertical_flip=True, horizontal_flip=True)             #ジェネレーターの設定
            gen = datagen.flow(image1, batch_size=1, save_to_dir=save_path, save_prefix='generated', save_format='jpg')       #ジェネレーターを作成
            
            
            for i in range(9):                  #range(x) x回繰り返すことでx枚の編集画像
                next(gen)
