import os

# Listing the path of all the celebA images
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def face_remap(shape):
  remapped_image = cv2.convexHull(shape)
  return remapped_image

def generate_mask():

	image_folder = 'CelebA/img_align_celeba/'
	save_folder = 'CelebA/output/'
	nomask_folder = 'CelebA/nomask/'
	
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)

	if not os.path.exists(nomask_folder):
		os.mkdir(nomask_folder)
		
	mask_count = 0
	list_img_full=listdir_fullpath('data/CelebA/img_align_celeba/') #change 

	for i, image_path in enumerate(list_img_full):