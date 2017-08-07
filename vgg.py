import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cosine as dcos
import cv2
from PIL import Image
import numpy as np
import math
import copy
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.models import model_from_json
from keras import backend as K
K.set_image_data_format( 'channels_last' )
import os
import csv
import shelve
import re

#load images from known folder
def image_folder():
	folder='known'
	return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

#detect face and crop image
crpimges = {}
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
	crpim = im.crop(box).resize((224,224))
	crpimges[imagePath] = crpim

#building VGGmodel
def convblock(cdim, nb, bits=3):
	L = []

	for k in range(1,bits+1):
		convname = 'conv'+str(nb)+'_'+str(k)
		#L.append( Convolution2D(cdim, 3, 3, border_mode='same', activation='relu', name=convname) ) # Keras 1
		L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) ) # Keras 2

	L.append( MaxPooling2D((2, 2), strides=(2, 2)) )

	return L

def vgg_face_blank():

	withDO = True 
	if True:
		mdl = Sequential()

		# First layer is a dummy-permutation = Identity to specify input shape
		mdl.add( Permute((1,2,3), input_shape=(224,224,3)) ) # WARNING : 0 is the sample dim

		for l in convblock(64, 1, bits=2):
			mdl.add(l)

		for l in convblock(128, 2, bits=2):
			mdl.add(l)

		for l in convblock(256, 3, bits=3):
			mdl.add(l)

		for l in convblock(512, 4, bits=3):
			mdl.add(l)

		for l in convblock(512, 5, bits=3):
			mdl.add(l)

		#mdl.add( Convolution2D(4096, 7, 7, activation='relu', name='fc6') ) # Keras 1
		mdl.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') ) # Keras 2
		if withDO:
			mdl.add( Dropout(0.5) )
		#mdl.add( Convolution2D(4096, 1, 1, activation='relu', name='fc7') ) # Keras 1
		mdl.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') ) # Keras 2
		if withDO:
			mdl.add( Dropout(0.5) )
		#mdl.add( Convolution2D(2622, 1, 1, name='fc8') ) # Keras 1
		mdl.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') ) # Keras 2
		mdl.add( Flatten() )
		mdl.add( Activation('softmax') )

		return mdl

	else:
		raise ValueError('not implemented')

facemodel = vgg_face_blank()


# load trained weight
# weights downloaded from http://www.vlfeat.org/matconvnet/pretrained/#face-recognition
data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
description = data['meta'][0,0].classes[0,0].description

def copy_mat_to_keras(kmodel):

	kerasnames = [lr.name for lr in kmodel.layers]

	# WARNING : important setting as 2 of the 4 axis have same size dimension
	#prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
	prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'

	for i in range(l.shape[1]):
		matname = l[0,i][0,0].name[0]
		if matname in kerasnames:
			kindex = kerasnames.index(matname)
			#print matname
			l_weights = l[0,i][0,0].weights[0,0]
			l_bias = l[0,i][0,0].weights[0,1]
			f_l_weights = l_weights.transpose(prmt)
			#f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
			#f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
			assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
			assert (l_bias.shape[1] == 1)
			assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
			assert (len(kmodel.layers[kindex].get_weights()) == 2)
			kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
			

copy_mat_to_keras(facemodel) # loading weights


# extract the second last layer feature vector
featuremodel = Model(inputs=facemodel.layers[0].input, outputs=facemodel.layers[-2].output)
def features(featmodel, crpimg, transform=False):


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



# calculate feature vectors for all known images and store it in dict
fvec = {}
for i in crpimges:
	arr = crpimges[i]
	f = features(featuremodel, arr, transform=True)
	fvec[i] = f

# store fvec as shelve file
shelf = shelve.open("known_vectors.shlf")
shelf["my_dict"] = fvec
shelf.close()

# store model and weights
model_json = featuremodel.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
featuremodel.save_weights("model.h5")

print("Saved model to disk")


print("success")


