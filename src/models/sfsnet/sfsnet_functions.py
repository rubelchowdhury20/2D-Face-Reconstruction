import os 
import numpy as np
from skipnet.skipnet_functions import * # TODO

def generate_list():
    # Function for the generation of image paths for training
    
    celebA_image_list = []
    celebA_normal_list = []
    celebA_albedo_list = []
    celebA_mask_list = []
    celebA_light_list = []

    if os.path.exists('/data/predicted_data_skipnet/'):

        folder_count = len(os.listdir('/data/predicted_data_skipnet/'))
    
    else:

        folder_count = 10

    # generating list for image, albedo, normal, lighting SH and mask in a similar order
    for i in range(0,folder_count):
        folder = str(i+1).zfill(4) # remove if not required
        print(folder)
        path = os.path.join('/data/predicted_data_skipnet/',folder)  # Need to change the path based on actual structure
        print(path)
        for root, _, filename in os.walk(path):
            filename.sort() 
            for each_file in filename:
                if (each_file.endswith(".png") and 'img' in each_file):
                    celebA_image_list.append(os.path.join(root, each_file))   
                elif (each_file.endswith(".png") and 'albedo' in each_file):
                    celebA_albedo_list.append(os.path.join(root, each_file))
                elif (each_file.endswith(".png") and 'normal' in each_file):
                    celebA_normal_list.append(os.path.join(root, each_file))
                elif (each_file.endswith(".png") and 'mask' in each_file):
                    celebA_mask_list.append(os.path.join(root, each_file))     
                elif (each_file.endswith(".txt")):
                    celebA_light_list.append(os.path.join(root, each_file))

    image_list = []
    normal_list = []
    albedo_list = []
    mask_list = []
    light_list = []
    if os.path.exists('data/synthetic_data/DATA_pose_15/'):
        folder_count = len(next(os.walk('data/synthetic_data/DATA_pose_15/'))[1])
    else:
        folder_count = 10

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

    # TODO: Generate the normal, albedo and mask list as output from the generate_syn_list itself. 
    # Setting the labels for synthetic data
    normal_label=[-1 for i in range(len(list(normal_list)))]
    albedo_label=[-1 for i in range(len(list(albedo_list)))]
    mask_label=[-1 for i in range(len(list(mask_list)))]

    # Setting the labels for real/celeba data
    celebA_normal_label=[1 for i in range(len(celebA_normal_list))]
    celebA_albedo_label=[1 for i in range(len(celebA_albedo_list))]
    celebA_mask_label=[1 for i in range(len(celebA_mask_list))]

    # Merge dataset for real and synthetic
    combine_image_list = []
    combine_normal_list = []
    combine_albedo_list = []
    combine_mask_list = []
    combine_light_list = []


    combine_image_list = list(image_list) + celebA_image_list
    combine_normal_list = list(normal_list) + celebA_normal_list
    combine_albedo_list = list(albedo_list) + celebA_albedo_list
    combine_mask_list = list(mask_list) + celebA_mask_list
    combine_light_list = list(light_list) + celebA_light_list

    # converting list to array 
    combine_image_list = np.asarray(combine_image_list)
    combine_albedo_list = np.asarray(combine_albedo_list)
    combine_normal_list = np.asarray(combine_normal_list)
    combine_light_list = np.asarray(combine_light_list)
    combine_mask_list = np.asarray(combine_mask_list)

    # Merge labels for real and synthetic
    combine_normal_label = []
    combine_albedo_label = []
    combine_mask_label = []

    combine_normal_label = normal_label + celebA_normal_label
    combine_albedo_label = albedo_label + celebA_albedo_label
    combine_mask_label = mask_label + celebA_mask_label

    # converting list to array 
    combine_normal_label = np.asarray(combine_normal_label)
    combine_albedo_label = np.asarray(combine_albedo_label)
    combine_mask_label = np.asarray(combine_mask_label)

    # Shuffling the entire dataset
    mapped = list(zip(combine_image_list, combine_mask_list, combine_normal_list, combine_albedo_list, combine_light_list, combine_normal_label, combine_albedo_label, combine_mask_label))
    np.random.shuffle(mapped)
    combined_list = np.array(mapped)
    print(combined_list.shape[0])

    features = np.transpose(np.asarray([combine_image_list, combine_mask_list]))
    labels = np.transpose(np.asarray([combine_normal_list, combine_albedo_list, combine_light_list, combine_normal_label, combine_albedo_label, combine_mask_label]))
    print(features.shape)
    assert features.shape[0] == labels.shape[0]

    return features, labels

def train_validation_split(features, labels):
    n_total = features.shape[0]
    train_max_index = int(n_total * 0.8)
    test_min_index = int((n_total * 0.8))

    train_features = np.transpose(features[:train_max_index])
    train_labels = np.transpose(labels[:train_max_index])
    test_features = np.transpose(features[test_min_index:])
    test_labels = np.transpose(labels[test_min_index:])

    train_data = {'features':train_features, 'labels':train_labels}
    test_data = {'features':test_features, 'labels':test_labels}

    return train_data, test_data
