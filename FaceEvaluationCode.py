
import cv2
import numpy as np
import numpy.random as npr
from PIL import Image, ImageDraw

import sys 
import dlib


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x 
    h = rect.bottom() - y 
    return (x, y, w, h)

def rect_to_gt(rect):
    l = rect.left()
    t = rect.top()
    r = rect.right() 
    b = rect.bottom() 
    return (l, t, r, b)

def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r)) 
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[1] + 1)
    xx1 = np.maximum(box[0], boxes[0])
    yy1 = np.maximum(box[1], boxes[1])
    xx2 = np.minimum(box[2], boxes[2])
    yy2 = np.minimum(box[3], boxes[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr
	
# main fuction begins
file_path="test_result.txt"
f=open(file_path, "w")
num = 0
iou = 0 
fzz = open('bbresult2.txt') # load the groundtruth here
lines = fzz.readline().split(' ')
print('/home/zz/CENG5050FaceDetectionEvaluation/Temp3_12Final/' + lines[0])
while lines:
    if len(lines[0])==0:
       break
    image = cv2.imread('/home/zz/CENG5050FaceDetectionEvaluation/Temp3_12Final/' + lines[0]) #you should change the path to find the test images
    print(lines[0])
    #predicting the bounding box result using your model, here using dlib.face_detector as an example. 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    gt = np.ones(4)
    prec = np.ones(4)
    gt[0] = int(float(lines[1]))
    gt[1] = int(float(lines[2]))
    gt[2] = int(float(lines[3]))
    gt[3] = int(float(lines[4]))

    # if has multiple outputs
    for (i, rect) in enumerate(rects):
       rect_pre = rect_to_gt(rect)
       prec[0] = rect_pre[0]
       prec[1] = rect_pre[1]
       prec[2] = rect_pre[2]
       prec[3] = rect_pre[3]
       #save the bounding box result in test_result.txt
       f.write(str(prec[0]))
       f.write(' ')
       f.write(str(prec[1]))
       f.write(' ')
       f.write(str(prec[2]))
       f.write(' ')
       f.write(str(prec[3]))
       f.write('\n')

       #compute the iou
       iou += IoU(gt, rect_pre)
       num += 1
       #debug
       print('num')
       print(num)
       print('IoU sum')
       print(iou)
       print(IoU(gt, rect_pre))
       

       (x, y, w, h) = rect_to_bb(rect)
       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

       cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    lines = fzz.readline().split(' ')
    # save the result img with predicted bounding box
    #cv2.imshow("Output", image)
    #cv2.imsave(".jpg", image)
    cv2.waitKey(0)

print('avg iou')
print(iou/num)
f.write(str(iou/num))

