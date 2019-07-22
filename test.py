# test script to preform prediction on test images inside 
# dataset/test/
#   -- image_1.jpg
#   -- image_2.jpg
#   ...

# organize imports
from __future__ import print_function

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import json
import pickle
import pandas as pd
import cv2
from imutils import paths

# load the trained logistic regression classifier
print ("[INFO] loading the classifier...")
classifier = pickle.load(open("hand_gesture.pickle", 'rb'))

base_model_vgg = VGG16(weights="imagenet")
model_vgg = Model(input=base_model_vgg.input, output=base_model_vgg.get_layer('fc1').output)
image_size_vgg = (224, 224)

# get all the train labels
train_labels = os.listdir("dataset")

cap = cv2.VideoCapture(1)



while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	#define region of interest
	roi=frame[100:250, 100:250]   
	roi_img = cv2.resize(roi,(224,224)) 
	x = np.expand_dims(roi_img, axis=0)
	x = preprocess_input(x)
	feature= model_vgg.predict(x)
	flat = feature.flatten()
	flat = np.expand_dims(flat, axis=0)
	predictions = classifier.predict_proba(np.atleast_2d(flat))[0]
	#predictions = np.argmax(predictions)
	preds = classifier.predict(flat)
	print(preds)
	print(predictions)
	#prediction = train_labels[preds[0]]

	cv2.rectangle(frame,(100,100),(350,350),(0,255,0),2) 
	cv2.putText(frame, "{}".format(preds), (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()