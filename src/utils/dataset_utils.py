import os
import gdown



def download_file():
	synthetic_path = "./data/synthetic_data"
	if not os.path.exists(synthetic_path):
	    os.makedirs(synthetic_path)


	synthetic_url = 'https://drive.google.com/uc?id=18YVh0idJ9tNgrYluDJqsE3oY_dWVIJBE'
	print("download started")
	gdown.download(synthetic_url, synthetic_path, quiet=False)
	print("download completed")