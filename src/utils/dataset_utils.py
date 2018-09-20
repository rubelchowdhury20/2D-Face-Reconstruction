import os
import requests
import tarfile
import zipfile


synthetic_path = "./data/synthetic_data/"
celeba_path = "./data/celeba_data/"
landmarks_path = "./data/landmarks/"


def load_dataset(synthetic_file_name, synthetic_file_id,
		celeba_file_name, celeba_file_id,
		mask_landmarks_name, mask_landmarks_id):
	
	download_files(synthetic_file_name, synthetic_file_id,
		celeba_file_name, celeba_file_id,
		mask_landmarks_name, mask_landmarks_id)
	unzip_files(synthetic_file_name, celeba_file_name, mask_landmarks_name)
	delete_zip_files(synthetic_file_name, celeba_file_name, mask_landmarks_name)

def download_files(synthetic_file_name, synthetic_file_id,
		celeba_file_name, celeba_file_id,
		mask_landmarks_name, mask_landmarks_id):

	if not os.path.exists(synthetic_path):
		os.makedirs(synthetic_path)
	if not os.path.exists(celeba_path):
		os.makedirs(celeba_path)
	if not os.path.exists(landmarks_path):
		os.makedirs(landmarks_path)


	#if(not os.path.exists(synthetic_path) || )
	print("Download is in progress for synthetic data...")
	download_file_from_google_drive(synthetic_file_id, synthetic_path + synthetic_file_name)
	print("Download is in progress for celeba data...")
	download_file_from_google_drive(celeba_file_id, celeba_path + celeba_file_name)
	print("Download is in progress for landmarks weights...")
	download_file_from_google_drive(mask_landmarks_id, landmarks_path + mask_landmarks_name)
	print("Donwload completed.")

def unzip_files(synthetic_file_name, celeba_file_name, mask_landmarks_name):
	print("Unzipping the tar file of synthetic data")
	tar = tarfile.open(synthetic_path + synthetic_file_name, "r:gz")
	tar.extractall(synthetic_path)
	tar.close()

	print("Unzipping the zip file of celeba data")
	zip_celeba = zipfile.ZipFile(celeba_path + celeba_file_name, "r")
	zip_celeba.extractall(celeba_path)
	zip_celeba.close()

def delete_zip_files(synthetic_file_name, celeba_file_name, mask_landmarks_name):
	os.remove(synthetic_path + synthetic_file_name)
	os.remove(celeba_path + celeba_file_name)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)