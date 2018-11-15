import numpy as np
import os
import cv2
import multiprocessing as mp
from copy import deepcopy
import time

## Functions to Aid on the Parallelization Task
def gaussian_smoothing(input_img):
								
	gaussian_filter=np.array([[0.109,0.111,0.109],
							  [0.111,0.135,0.111],
							  [0.109,0.111,0.109]])
								
	return cv2.filter2D(input_img,-1,gaussian_filter) 

def canny_edge_detection(input):
	
	input = input.astype('uint8')

	# Using OTSU thresholding - bimodal image
	otsu_threshold_val, ret_matrix = cv2.threshold(input,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	#lower_threshold = otsu_threshold_val * 0.8
	#upper_threshold = otsu_threshold_val * 1.7
	
	lower_threshold = otsu_threshold_val * 0.4
	upper_threshold = otsu_threshold_val * 1.3
	
	print(lower_threshold,upper_threshold)
	
	#print(lower_threshold,upper_threshold)
	edges = cv2.Canny(input, lower_threshold, upper_threshold)
	return edges

def HoughCircles(input,circles): 
	rows = input.shape[0] 
	cols = input.shape[1] 
	
	# initializing the angles to be computed 
	sinang = dict() 
	cosang = dict() 
	
	# initializing the angles  
	for angle in range(0,360): 
		sinang[angle] = np.sin(angle * np.pi/180) 
		cosang[angle] = np.cos(angle * np.pi/180) 
			
	# initializing the different radius
	# For Test Image <----------------------------PLEASE SEE BEFORE RUNNING------------------------------->
	#radius = [i for i in range(10,70)]
	#For Generic Images
	length=int(rows/2)
	radius = [i for i in range(5,length)]

	
	# Initial threshold value 
	threshold = 190 
	
	for r in radius:
		#Initializing an empty 2D array with zeroes 
		acc_cells = np.full((rows,cols),fill_value=0,dtype=np.uint64)
		 
		# Iterating through the original image 
		for x in range(rows): 
			for y in range(cols): 
				#print (x)
				#print (y)
				if input[x][y] == 255:# edge 
					# increment in the accumulator cells 
					for angle in range(0,360): 
						b = int(y - int(round(r * sinang[angle])))
						a = int(x - int(round(r * cosang[angle])))
						if a >= 0 and a < rows and b >= 0 and b < cols: 
							acc_cells[a][b] += 1
							 
		print('For radius: ',r)
		acc_cell_max = np.amax(acc_cells)
		print('max acc value: ',acc_cell_max)
		
		if(acc_cell_max > 150):  

			print("Detecting the circles for radius: ",r)	   
			
			# Initial threshold
			acc_cells[acc_cells < 150] = 0  
			   
			# find the circles for this radius 
			for i in range(rows): 
				for j in range(cols): 
					if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j] >= 150):
						avg_sum = np.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
						#print("Intermediate avg_sum: ",avg_sum)
						if(avg_sum >= 33):
							#print("For radius: ",r,"average: ",avg_sum,"\n")
							circles.append((i,j,r))
							acc_cells[i:i+5,j:j+7] = 0




# Foi mal o copypaste aqui negada..
# Tomar cuidado com essa funcao.
def applyHoughCircleNoFilter(imgFile, p2, minR, maxR):
	# Get First Image
	imagePath = os.path.join('images', imgFile)
	img = cv2.imread(imagePath,0)

	# Show image
	#cv2.imshow('Image: ' + imgFile,img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
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

	cv2.imwrite(os.path.join('resultImagesSeq', imgFile), cimg)
	#cv2.imshow('Image: ' + imgFile + ' with its detected circles.',cimg)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def applyHoughCircle(imgFile, p2, minR, maxR):
	# Get First Image
	imagePath = os.path.join('images', imgFile)
	img = cv2.imread(imagePath,0)

	# Show image
	#cv2.imshow('Image: ' + imgFile,img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
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

	cv2.imwrite(os.path.join('resultImagesSeq', imgFile), cimg)
	#cv2.imshow('Image: ' + imgFile + ' with its detected circles.',cimg)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()



def applyHoughCircleParallel(imgFile):
	# Get First Image
	imagePath = os.path.join('images', imgFile)
	orig_img = cv2.imread(imagePath)

	input = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)

	# Create copy of the orignial image
	input_img = deepcopy(input)
	
	#Steps
	#1. Denoise using Gaussian filter and detect edges using canny edge detector
	smoothed_img = gaussian_smoothing(input_img)
	
	#2. Detect Edges using Canny Edge Detector
	edged_image = canny_edge_detection(smoothed_img)
	#3. Detect Circle radius
	#4. Perform Circle Hough Transform
	circles = []
	
	# cv2.imshow('Circle Detected Image',edged_image)
   
	# Detect Circle 
	HoughCircles(edged_image,circles)  
	
	# Print the output
	for vertex in circles:
		cv2.circle(orig_img,(vertex[1],vertex[0]),vertex[2],(0,255,0),1)
		cv2.rectangle(orig_img,(vertex[1]-2,vertex[0]-2),(vertex[1]-2,vertex[0]-2),(0,0,255),3)
	
	cv2.imshow('Circle Detected Image',orig_img) 
	cv2.imwrite(os.path.join('resultImagesSeq', imgFile), orig_img)






## TODO: Measure time for each case and append to file
def sequencialHoughAppliance ():
	applyHoughCircle ('bacteria.jpg', 30, 10, 25)
	applyHoughCircle ('bacteria2.jpg', 30, 0, 40)
	applyHoughCircle ('3TennisBalls.jpg', 70, 0, 0)
	applyHoughCircle ('AerialViewTrees.jpg', 20, 5, 40)
	applyHoughCircle ('ForgedAerialViewTrees.jpg', 30, 10, 60)
	applyHoughCircleNoFilter ('MapRoundaboutView.jpg', 40, 0, 0)
	applyHoughCircleNoFilter ('MapView.jpg', 17, 0, 10)




start = time.time()
sequencialHoughAppliance()
end = time.time()
print(end-start)
