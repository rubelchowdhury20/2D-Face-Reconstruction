import argparse

from src import utils

def main(args):
	utils.load_dataset(
		args.synthetic_file_name, args.synthetic_file_id,
		args.celeba_file_name, args.celeba_file_id,
		args.mask_landmarks_name, args.mask_landmarks_id)

	utils.generate_mask(args.skipnet_batch_size)

if __name__ == '__data_loader__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--skipnet_batch_size",
		type=int,
		default=10,
		help="Skipnet batch size")


	main(parser.parse_args())