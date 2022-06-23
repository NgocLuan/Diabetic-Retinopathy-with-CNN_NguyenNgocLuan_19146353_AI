from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import scale
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import tqdm
from tqdm import trange
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__, template_folder='templates')

dic = {0 : 'MẮC BỆNH NHẸ', 1 : 'MẮC BỆNH VỪA',2 : 'KHỎE MẠNH', 3 : 'TĂNG SINH',4 : 'MẮC BỆNH NẶNG'}

model = load_model('EfficientNetB4_MODEL2.h5')

model.make_predict_function()



def predict_label(img_path_2):
	i = tf.keras.utils.load_img(img_path_2, target_size=(224,224))
	i = tf.keras.utils.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	i = i.astype('float32')
	
	p = np.argmax(model.predict(i), axis=-1)  
	# p = model.predict_classes(i)
	return dic[p[0]]

def Btgraham(img_1, scale):

	a=cv2.addWeighted(img_1, 4, cv2.GaussianBlur(img_1, (0,0) ,10) , -4 ,128)
	# b=np.zeros(a.shape)
	# cv2.circle(b ,(math.floor(a.shape[1]/2),math.floor(a.shape[0]/2)) , int(scale*1), (1, 1, 1) , -1, 8, 0)
	# a=a*b+ 128*(1-b)
	return a.astype("uint8")

# routes
@app.route("/", methods=['GET', 'POST'])
def main():

	return render_template("index.html")

@app.route("/about")
def about_page():
	return "NGUYEN NGOC LUAN - 19146353..!!!"

@app.route("/submit", methods = ['GET', 'POST'])



#bt_img = Btgraham(crop_img, 300)
#display_img(2,1 ,img = [crop_img, bt_img], title = ["Crop", "Btgraham"])
#cv2.imwrite("/content/drive/MyDrive/data_project/processed_1.jpg",bt_img)

def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)

		img_1 = cv2.imread(img_path)

		bt_img = Btgraham(img_1, scale)
		img_path_2 = "bt/" + img.filename
		#img.save(img_path_2, bt_img)
		cv2.imwrite("bt/" + img.filename , bt_img)
		p = predict_label(img_path_2)

	return render_template("index.html", prediction = p, img_path = img_path)




if __name__ =='__main__':
	#app.debug = True
    #app.run(host='127.0.0.1',port=5500)
	app.run(debug=True)
    
	