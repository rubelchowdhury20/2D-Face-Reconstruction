import argparse

from src import utils

def main(args):
	utils.load_dataset(
		args.synthetic_file_name, args.synthetic_file_id,
		args.celeba_file_name, args.celeba_file_id,
		args.mask_landmarks_name, args.mask_landmarks_id)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--synthetic_file_name",
		type=str,
		default="Syn_data.tar.gz",
		help="File name for Synthetic Data")
	parser.add_argument(
		"--synthetic_file_id",
		type=str,
		default="18YVh0idJ9tNgrYluDJqsE3oY_dWVIJBE",
		help="Google Drive id for Synthetic Data")
	parser.add_argument(
		"--celeba_file_name",
		type=str,
		default="img_align_celeba.zip",
		help="File name for Celeba Data")
	parser.add_argument(
		"--celeba_file_id",
		type=str,
		default="0B7EVK8r0v71pZjFTYXZWM3FlRnM",
		help="Google Drive id for Celeba Data")
	parser.add_argument(
		"--mask_landmarks_name",
		type=str,
		default="shape_predictor_68_face_landmarks.dat",
		help="File name for Mask Landmarks weights")
	parser.add_argument(
		"--mask_landmarks_id",
		type=str,
		default="1NjkXxViYxZF1-xB_3mfrTEyKTYqLea8s",
		help="Google Drive id for Mask Landmarks weights")

	main(parser.parse_args())