import shutil
import os
from scipy.misc.pilutil import imsave
import tensorflow as tf
import numpy as np
from .sfsnet_arch import *
from .sfsnet_functions import *


def save_predict_output(estimator, predict_input):
	folder_count_celeba = 0
	image_count = 0
	images_in_folder = 300
	predicted_images_path = "data/sfsnet_inference/"
	current_folder_path = ""

	if not os.path.exists(predicted_images_path):
		os.makedirs(predicted_images_path)
	else:
		shutil.rmtree(predicted_images_path)
		os.makedirs(predicted_images_path)


	for i in estimator.predict(input_fn=predict_input):
		if(image_count % images_in_folder == 0):
			folder_count_celeba = folder_count_celeba + 1
			current_folder_path = predicted_images_path + '/' + str(folder_count_celeba).zfill(4)
			os.makedirs(current_folder_path)
		img_path = current_folder_path + '/' + str(image_count).zfill(6) + '_img.png'
		mask_path = current_folder_path + '/' + str(image_count).zfill(6) + '_mask.png'
		normal_path = current_folder_path + '/' + str(image_count).zfill(6) + '_normal.png'
		albedo_path = current_folder_path + '/' + str(image_count).zfill(6) + '_albedo.png'
		light_path = current_folder_path + '/' + str(image_count).zfill(6) + '_light.txt'
		imsave(img_path, i["image"])
		imsave(mask_path, i["mask"])
		imsave(normal_path, i["normal"])
		imsave(albedo_path, i["albedo"])
		light_array = '\t'.join(str(e) for e in i["light"])
		with open(light_path, 'w+') as file:
			file.write(light_array)
		image_count = image_count + 1


def predict(test_data, batch_size, learning_rate):
	predict_input = predict_input_function(lambda: input_fn(test_data, batch_size=batch_size))
	predictions = list(estimator.predict(input_fn=predict_input_fn))