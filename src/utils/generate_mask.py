import os
import cv2
import dlib
import shutil
import numpy as np
import skimage
from skimage import io
import imutils
from imutils import face_utils

# Listing the path of all the celebA images
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def face_remap(shape):
  remapped_image = cv2.convexHull(shape)
  return remapped_image

def generate_mask():

	image_folder = 'data/celeba_data/img_align_celeba/'
	save_folder = 'data/celeba_data/celeba_mask/'
	nomask_folder = 'data/celeba_data/celeba_nomask/'
	
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)

	if not os.path.exists(nomask_folder):
		os.mkdir(nomask_folder)
		
	mask_count = 0
	list_img_full=listdir_fullpath('data/celeba_data/img_align_celeba/') 

	for i, image_path in enumerate(list_img_full):

		if(mask_count == (len(list_img_full) - (len(list_img_full) % skipnet_batch_size))):
			break
		# read image
		image = io.imread(image_path)
		
		if i%500 == 0:
			print("Mask generation for "+ str(i) +" images are done")

		name = image_path.strip().split('/')[-1]
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	  

		out_face = np.zeros((image.shape[0], image.shape[1]))
		SHAPE_PREDICTOR = 'data/landmarks/shape_predictor_68_face_landmarks.dat'
		# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

		# detect faces in the grayscale image
		rects = detector(gray, 1)
		mask_exists = False
		
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			mask_count+=1
		"""
		Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
		"""
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#initialize mask array
		remapped_shape = np.zeros_like(shape)

		feature_mask = np.zeros((image.shape[0], image.shape[1]))

		# we extract the face
		remapped_shape = face_remap(shape)
		cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
		feature_mask = feature_mask.astype(np.bool)
		out_face[feature_mask] = gray[feature_mask]
		out_face = out_face * 255
		out_face = out_face.reshape(out_face.shape[0], out_face.shape[1], 1)
		out_face = np.tile(out_face, (1, 1, 3))
		cv2.imwrite(os.path.join(save_folder, name), out_face)
		mask_exists = True
		if mask_exists == False :
			mask_count-=1
			shutil.move(image_path, nomask_folder + "/" + name)		#create
			print("Mask is not created for "+name+". Image is moved out!")
