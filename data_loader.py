import argparse

from src import utils

def main(args):
	print("hello")
	utils.load_dataset()
	utils.generate_mask(args.skipnet_batch_size)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--skipnet_batch_size",
		type=int,
		default=10,
		help="Skipnet batch size")


	main(parser.parse_args())