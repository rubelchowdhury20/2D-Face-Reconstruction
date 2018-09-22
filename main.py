import argparse

from src import utils
from src.models import skipnet


def main(args):
	utils.load_dataset(
		args.synthetic_file_name, args.synthetic_file_id,
		args.celeba_file_name, args.celeba_file_id,
		args.mask_landmarks_name, args.mask_landmarks_id)

	utils.generate_mask(args.skipnet_batch_size)
	skipnet.train(args.skipnet_batch_size, args.skipnet_learning_rate, args.skipnet_epochs)



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
	parser.add_argument(
		"--skipnet_batch_size",
		type=int,
		default=10,
		help="Skipnet batch size"
		)
	parser.add_argument(
		"--skipnet_learning_rate",
		type=float,
		default=0.00001,
		help="Skipnet learning rate")
	parser.add_argument(
		"--skipnet_epochs",
		type=int,
		default=10,
		help="Skipnet total epochs")

	main(parser.parse_args())