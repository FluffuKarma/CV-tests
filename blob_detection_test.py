import cv2
import numpy as np

#webcam
webcam = cv2.VideoCapture(0) 

while True:

	# read image
	(_,img) = webcam.read()

	# convert img to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
      
	# Threshold of blue in HSV space 
	lower_blue = np.array([30,150,50]) 
	upper_blue = np.array([255,255,180]) 
  
	# preparing the mask to overlay 
	mask = cv2.inRange(hsv, lower_blue, upper_blue) 
      
	# The black region in the mask has the value of 0, 
	# so when multiplied with original image removes all non-blue regions 
	result = cv2.bitwise_and(img, img, mask = mask) 
  
	cv2.imshow('frame', img) 
	cv2.imshow('mask', mask) 
	cv2.imshow('result', result) 

	# # do adaptive threshold on gray image
	# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)

	# # apply morphology open then close
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
	blob = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
	blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

	# invert blob
	#blob = (255 - blob)

	# Get contours
	cnts = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	# big_contour = max(cnts, key=cv2.contourArea)

	good_cnts=[]

	# test blob size
	blob_area_thresh_min = 600
	blob_area_thresh_max = 30000
	# blob_area = cv2.contourArea(big_contour)
	# if blob_area < blob_area_thresh:
	#     print("Blob Is Too Small")
	for i in cnts:
		if cv2.contourArea(i)>blob_area_thresh_min and cv2.contourArea(i)<blob_area_thresh_max:
			good_cnts.append(i)

	# draw contour
	result = img.copy()
	cv2.drawContours(result, good_cnts, -1, (0,0,255), 1)

	# write results to disk
	# cv2.imwrite("doco3_threshold.jpg", thresh)
	# cv2.imwrite("doco3_blob.jpg", blob)

	# display it
	#cv2.imshow("IMAGE", img)
	#cv2.imshow("THRESHOLD", thresh)
	cv2.imshow("BLOB", blob)
	cv2.imshow("RESULT", result)
	
	key = cv2.waitKey(10) 
	if key == 27: 
		break

cv2.destroyAllWindows() 
webcam.release() 