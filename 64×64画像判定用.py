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

model = load_model('akaten_learn_model_MD00195.h5')
kanri = 0
count = 1

#カメラ撮影保存関数
def camera(count):
	camera = PiCamera()				#Picamera起動
	camera.resolution = (784, 784)	#撮影画像サイズ
	camera.start_preview()			#カメラから映像取得開始
	sleep(5)						#5秒待つ
	camera.capture('/home/pi/hantei/hantei'+str(count)+'.jpg')	#映像を画像として保存
	camera.stop_preview()			#カメラの映像取得終了
	camera.close()					#camera終了


def create_image(image):
	#filepath = "/home/pi/input" + "/" + file        #読み込むディレクトリ(フォルダ)を指定
        #image = cv2.imread(filepath)           #指定されたファイルを読み込む
	#image = image[150:400, 250:500]           #画像のカット(ほしい対象物でない部分をカットしている)　全部必要なら編集して飛ばす
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#画像をグレースケールに変換
	preprocessed = cv2.GaussianBlur(gray, (5, 5), 0)	#画像のぼかし

	#            画像の閾値処理
	_, origin_binary = cv2.threshold(preprocessed, 110, 255, cv2.THRESH_BINARY)

	# 色の反転
	#coins_binary = cv2.bitwise_not(coins_binary)

	# 輪郭検出
	_, origin_contours, _ = cv2.findContours(origin_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
	# 元の画像をコピー
	origin_and_contours = np.copy(image)

	# 何個かあるかカウント
	min_coin_area = 60
	large_contours = [cnt for cnt in origin_contours if cv2.contourArea(cnt) > min_coin_area]

	# 輪郭を描写　（引数　＝＝＝　1：入力画像	2：listとして保存されている輪郭	3：listの何番目の輪郭か（−1なら全部）	4~：輪郭線の情報）
	origin_contour = cv2.drawContours(origin_and_contours, large_contours, -1, (255,0,0))
	#cv2.imshow("origin_contours", origin_contour)
	#cv2.waitKey(0)
	# カウントした個数の表示
	#print('number of coins: %d' % len(large_contours))


	# 元の画像をコピー
	bounding_img = np.copy(image)
	#cv2.imshow('image',bounding_img)
	#cv2.waitKey(0)
	# 外接矩形の描写
	for contour in large_contours:
		x, y, w, h = cv2.boundingRect(contour)
		gaisetu = cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 0, 0), 1)

	i = 1
	#外接矩形で切り取る
	for contour in large_contours:
		x, y, w, h = cv2.boundingRect(contour)
		gaisetu = cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 0, 0), 1)
		crop = gaisetu[y:y+h, x:x+w]
		#print(crop.shape)
		i = i + 1
                  
	gousei = cv2.imread('/home/pi/gazou/kuro.jpg')
	gousei2 = crop
	#cv2.imshow('image',gousei2)
	#cv2.imshow('haikei',gousei)
	#cv2.waitKey(0)
	#print(gousei.shape)

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
	return gousei




#撮影用のwhileループ
while kanri == 0:
	print('判定を開始します。　判定物を置いて y を入力してください。　判定を終了する場合は n　を入力してください。')
	yes_no = input()
	
	if yes_no == 'y':		#入力がyなら
		camera(count)		#カメラ撮影用の関数を起動
		image = cv2.imread('/home/pi/hantei/hantei'+str(count)+'.jpg')
		
		image = create_image(image)
       
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
		"""
		#img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)		#opencvはBGRなのでRGBに変更　画層表示の際等に齟齬が生じる可能性があるのでとりあえずRGBにしておく
		image = img_masked	#マスク処理した画像でimageを上書き
		image = cv2.resize(image, (64, 64))		#画像の縮小
		cv2.imshow('hantei', image)
		cv2.waitKey(0)
		cv2.imwrite('/home/pi/hantei/hanteigazou'+str(count)+'.jpg', image)
		cv2.destroyAllWindows()
		#---------------------------------------------------------------------------------
		result = model.predict_classes(np.array([image / 255.]))
		result2 = model.predict(np.array([image / 255.]))
        
		print(result2)
		if result[0] == 0:
			print('赤点ありと判定しました')
		else:
			print('赤点なしと判定しました')
		
		count =count + 1	#countを+1する
		
		
	elif yes_no == 'n':		#入力がnなら
		kanri = 1			#ループ条件崩す
	else:
		print('入力が間違っています')
		
sys.exit()
