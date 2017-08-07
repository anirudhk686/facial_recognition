import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cosine as dcos
import cv2
from PIL import Image
import numpy as np
import math
import copy
import pandas as pd
from keras.models import model_from_json
from keras import backend as K
K.set_image_data_format( 'channels_last' )
import os
import csv
import re
from scipy.spatial.distance import cosine
import shelve


# load image from unknown folder
def image_folder():
	folder='unknown'
	return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

#detect face and crop image
for image in image_folder():
	imagePath = image
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	# Read the image
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray, 1.2, 5)

	im = Image.open(imagePath)

	(x, y, w, h) = faces[0]
	center_x = x+w/2
	center_y = y+h/2
	b_dim = min(max(w,h)*1.2,im.width, im.height) # WARNING : this formula in incorrect
	#box = (x, y, x+w, y+h)
	box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)
	# Crop Image
	crpimg = im.crop(box).resize((224,224))

#load model and weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#calculate feature vector of unkmown image
def features(featmodel, crpimg, transform=False):
	# transform=True seems more robust but I think the RGB channels are not in right order

	imarr = np.array(crpimg).astype(np.float32)

	if transform:
		imarr[:,:,0] -= 129.1863
		imarr[:,:,1] -= 104.7624
		imarr[:,:,2] -= 93.5940
		
	imarr = np.expand_dims(imarr, axis=0)

	fvec = featmodel.predict(imarr)[0,:]
	# normalize
	normfvec = math.sqrt(fvec.dot(fvec))
	return fvec/normfvec

f = features(loaded_model, crpimg, transform=True)

# get known feature vectors from shelve file
shelf = shelve.open("known_vectors.shlf")
known = shelf["my_dict"]
shelf.close()

# compare known vs unknow to get the best match
value =1
for i in known:
	sim = cosine(known[i],f)
	if sim<value:
		value = sim
		match = i
print(match,1-value)

