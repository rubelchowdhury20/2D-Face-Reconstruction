import os
import numpy as np
from matplotlib import path

def generate_syn_name_list():
	image_list = []
	normal_list = []
	albedo_list = []
	mask_list = []
	light_list = []
	folder_count = len(next(os.walk('data/synthetic_data/DATA_pose_15/'))[1])

	# generating list for image, albedo, normal, lighting SH and mask in a similar order
	for i in range(0,folder_count):
		folder = str(i+1).zfill(4)
		path = 'data/synthetic_data/DATA_pose_15/' + folder + '/'
		addrs = os.walk(path)
		for root, dirs, filename in addrs:
			filename.sort() 
			for file in filename:
				if (file.endswith(".png") and 'face' in file):
					image_list.append(os.path.join(root, file))       
				elif (file.endswith(".png") and 'albedo' in file):
					albedo_list.append(os.path.join(root, file))
				elif (file.endswith(".png") and 'normal' in file):
					normal_list.append(os.path.join(root, file))
				elif (file.endswith(".png") and 'mask' in file):
					mask_list.append(os.path.join(root, file))         
				elif (file.endswith(".txt")):
					light_list.append(os.path.join(root, file))

	# converting list to numpy array 
	image_list = np.asarray(image_list)
	albedo_list = np.asarray(albedo_list)
	normal_list = np.asarray(normal_list)
	light_list = np.asarray(light_list)
	mask_list = np.asarray(mask_list)

	features = np.transpose(np.asarray([image_list, mask_list]))
	labels = np.transpose(np.asarray([normal_list, albedo_list, light_list]))

	return features, labels

def train_validation_split(features, labels):
	# Assigning the index values for train and validation data
	n_total = features.shape[0]
	train_min_index = int(n_total * 0)
	train_max_index = int(n_total * 0.8)
	test_min_index = int((n_total * 0.8))
	test_max_index= int(n_total * 1)

	train_features = features[:train_max_index]
	train_labels = labels[:train_max_index]
	test_features = features[test_min_index:]
	test_labels = labels[test_min_index:]

	return train_features, train_labels, test_features, test_labels

