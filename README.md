
#Useful framework for evaluate model with bounding box output in IOU metric  
Here it is used for face detection model evaluation in IOU metric

Demo description:
This demo is used to test the face detection pre-trained model according to IOU score  
(1) Download the following necessary items and link them in FaceEvaluationCode.py:  

<1> Pre-trained model: shape_predictor_68_face_landmarks.dat (https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2)  
<2> link Data img Temp3_12Final/ For example, I put the file in Temp3_12Final/  
<3> Ground truth bounding box file is bbresult2.txt  
  

(2) The required library files include:  
  
Import cv2  
Import numpy as np  
Import numpy.random as npr  
Import Image from Image, ImageDraw  
Import system  
Import dlib  
  
(3) Finally run:  
Python FaceEvaluationCode.py  
  
you will find your average IOU score in test_result.txt, For simplifing, here each test face image   only contain one face  
  
