import sys
from picamera import PiCamera
from time import sleep

#whileループ管理用変数
kanri = 0
#写真番号管理用変数
count = 0
#撮影続行管理用変数
yes_no = 0

#カメラ撮影保存関数
def camera(count):
	camera = PiCamera()				#Picamera起動
	camera.resolution = (784, 784)	#撮影画像サイズ
	camera.start_preview()			#カメラから映像取得開始
	sleep(5)						#5秒待つ
	camera.capture('/home/pi/camera/image'+str(count)+'.jpg')	#映像を画像として保存
	camera.stop_preview()			#カメラの映像取得終了
	camera.close()					#camera終了

#保存番号の要求
print('写真保存用番号を入力してください。\n 写真はimageX.jpgで保存されます。 X = ')
count = input()	#入力をcountに入れる
count = int(count)	#入力のままだと文字列扱いなのでint型に変更

#撮影用のwhileループ
while kanri == 0:
	print('撮影しますか？　撮影をする場合は y 　撮影を終了する場合は n　を入力してください')
	yes_no = input()
	
	if yes_no == 'y':		#入力がyなら
		camera(count)		#カメラ撮影用の関数を起動
		print('写真 image'+str(count)+'jpg　が保存されました')
		count =count + 1	#countを+1する
		
	elif yes_no == 'n':		#入力がnなら
		kanri = 1			#ループ条件崩す
	else:
		print('入力が間違っています')
		
sys.exit()
