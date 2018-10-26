import numpy as np
import os
import cv2
import multiprocessing as mp
import psutil

# Foi mal o copypaste aqui negada..
# Tomar cuidado com essa função.
def applyHoughCircleNoFilter(imgFile, p2, minR, maxR):
	# Get First Image
	imagePath = os.path.join('images', imgFile)
	img = cv2.imread(imagePath,0)

	# Show image
	cv2.imshow('Image: ' + imgFile,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

	#void cv::HoughCircles (	InputArray 	image,
	#						int 	method,
	#						double 	dp,
	#						double 	minDist,
	#						double 	param1 = 100,
	#						double 	param2 = 100, ## Related to Precision
	#						int 	minRadius = 0,
	#						int 	maxRadius = 0 
	#						)
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
	                            param1=50, param2=p2, minRadius=minR, maxRadius=maxR)


	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
		    # draw the outer circle
		    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
		    # draw the center of the circle
		    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('Image: ' + imgFile + ' with its detected circles.',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def applyHoughCircle(imgFile, p2, minR, maxR):
	# Get First Image
	imagePath = os.path.join('images', imgFile)
	img = cv2.imread(imagePath,0)

	# Show image
	cv2.imshow('Image: ' + imgFile,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	## Perform Gaussian smoothing to reduce sharp edges and noise.
	img = cv2.GaussianBlur(img,(9,9),0)
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

	#void cv::HoughCircles (	InputArray 	image,
	#						int 	method,
	#						double 	dp,
	#						double 	minDist,
	#						double 	param1 = 100,
	#						double 	param2 = 100, ## Related to Precision
	#						int 	minRadius = 0,
	#						int 	maxRadius = 0 
	#						)
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
	                            param1=50, param2=p2, minRadius=minR, maxRadius=maxR)


	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
		    # draw the outer circle
		    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
		    # draw the center of the circle
		    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('Image: ' + imgFile + ' with its detected circles.',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


#applyHoughCircle ('bacteria.jpg', 30, 10, 25)
#applyHoughCircle ('bacteria2.jpg', 30, 0, 40)
#applyHoughCircle ('3TennisBalls.jpg', 70, 0, 0)
#applyHoughCircle ('AerialViewTrees.jpg', 20, 5, 40)
#applyHoughCircle ('ForgedAerialViewTrees.jpg', 30, 10, 60)
#applyHoughCircleNoFilter ('MapRoundaboutView.jpg', 40, 0, 0)
#applyHoughCircleNoFilter ('MapView.jpg', 17, 0, 10)











