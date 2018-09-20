import os

def skipnet_preprocessing():
	pass

def generate_file_list():
	image_list = []
	normal_list = []
	albedo_list = []
	mask_list = []
	light_list = []
	folder_count = len(next(os.walk('data/synthetic_data/DATA_pose_15/'))[1])

	# generating list for image, albedo, normal, lighting SH and mask in a similar order
	for i in range(0,folder_count_skip):
	folder = str(i+1).zfill(4)
	print(folder)
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
