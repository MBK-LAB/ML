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
            filepath = "/home/pi/input"  + "/" + file        #読み込むディレクトリ(フォルダ)を指定
            image = cv2.imread(filepath)           #指定されたファイルを読み込む
            image = image[0:700, 0:700]           #画像のカット(ほしい対象物でない部分をカットしている)　全部必要なら編集して飛ばす
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
            preprocessed = cv2.GaussianBlur(gray, (5, 5), 0)	#画像のぼかし

            #画像の閾値処理
            _, origin_binary = cv2.threshold(preprocessed, 50, 255, cv2.THRESH_BINARY)

            # 色の反転
            #coins_binary = cv2.bitwise_not(coins_binary)

            # 輪郭検出
            _, origin_contours, _ = cv2.findContours(origin_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
            # 元の画像をコピー
            origin_and_contours = np.copy(image)

            # 何個かあるかカウント
            min_coin_area = 700
            large_contours = [cnt for cnt in origin_contours if cv2.contourArea(cnt) > min_coin_area]
 
            # 輪郭を描写　（引数　＝＝＝　1：入力画像	2：listとして保存されている輪郭	3：listの何番目の輪郭か（−1なら全部）	4~：輪郭線の情報）
            origin_contour = cv2.drawContours(origin_and_contours, large_contours, -1, (255,0,0))
            #cv2.imshow("origin_contours", origin_contour)
            #cv2.waitKey(0)
            # カウントした個数の表示
            print('number of coins: %d' % len(large_contours))


            # 元の画像をコピー
            bounding_img = np.copy(image)
            #cv2.imshow('image',bounding_img)
            #cv2.waitKey(0)
            # 外接矩形の描写
            for contour in large_contours:
                  x, y, w, h = cv2.boundingRect(contour)
                  gaisetu =cv2.rectangle(bounding_img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            #cv2.imshow("gaisetu", gaisetu)
            #cv2.waitKey(0)
            
            
            
            i = 1
            #外接矩形で切り取る
            for contour in large_contours:
                  x, y, w, h = cv2.boundingRect(contour)
                  gaisetu = cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                  crop = gaisetu[y:y+h, x:x+w]
                  print(crop.shape)
                  i = i + 1
            
            
            
            if crop.shape[0] > 150 or crop.shape[1] >150:
                  haikei_path = '/home/pi/gazou/背景集/kuro400.jpg'
            else:
                  haikei_path = '/home/pi/gazou/背景集/kuro.jpg'
                  
            gousei = cv2.imread(haikei_path)
            gousei2 = crop
            #cv2.imshow('image',gousei2)
            #cv2.imshow('haikei',gousei)
            #cv2.waitKey(0)
            print(gousei2.shape[0])

            gousei_size = gousei.shape[:2]
            hantei_size = gousei2.shape[:2]

            if hantei_size[0] > gousei_size[0] or hantei_size[1] > gousei_size[1]:
                  raise Exception("img is larger than size")

            row = (gousei_size[1] - hantei_size[0]) // 2
            col = (gousei_size[0] - hantei_size[1]) // 2
            gousei[row:(row + hantei_size[0]), col:(col + hantei_size[1])] = gousei2
            #cv2.imshow("white2", gousei)
            #cv2.waitKey(0)
      
            """
            cv2.imshow("coins", image)          #カットした画像確認用の画像表示
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            image = cv2.cvtColor(gousei, cv2.COLOR_BGR2RGB)        #opencvで読み込むとBGRデータなのでRGBに直しておく
            #print(image.sfor contour in large_contours:
            rect = cv2.minAreaRect(contour)
            center, size, angle = rect
            size = int(np.round(size[0])),int(np.round(size[1])) 
            box = cv2.boxPoints(rect)
            print(box)
            box = np.int0(box)
            gaisetu = cv2.drawContours(bounding_img,[box],0,(0,0,0 ),2)
            #cv2.imshow('aa',gaisetu)
            #cv2.waitKey(0)
            M = cv2.getAffineTransform(np.float32(box[1:4]),  np.float32([[0, 0], [size[0], 0], [size[0], size[1]]]))
            crop = cv2.warpAffine(bounding_img, M, size)
            #cv2.imshow('crop', crop)hape)       #imageの形を表示　（size,size,channel数）で表示
            #cv2.imshow('image', image)
            #cv2.waitKey(0)
            """
            image1 = image1.convert("RGB")            #opencv使わない時はこっち
            image1 = image1.resize((32, 32))
            image = Image.open(filepath)
            """
            
            print(filepath)               #現在のファイルを表示
            image1 = np.array(image)      #imageをnumpyに変更
            print(image1.shape)
            #image1 = cv2.resize(image1, (64, 64))
            #cv2.imshow('image', image1)
            #cv2.waitKey(0)
            
            image1 = image1[np.newaxis]    #image1を4次元情報に変更 
            #datagen = ImageDataGenerator(rotation_range=90, vertical_flip=True, horizontal_flip=True)             #ジェネレーターの設定
            datagen = ImageDataGenerator(rotation_range=5)             #ジェネレーターの設定
            gen = datagen.flow(image1, batch_size=1, save_to_dir=save_path, save_prefix='generated', save_format='jpg')       #ジェネレーターを作成
            
            
            for i in range(1):                  #range(x) x回繰り返すことでx枚の編集画像
                next(gen)
                
