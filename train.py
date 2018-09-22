import argparse

from src import utils
from src.models import skipnet


def main(args):
	skipnet.train(args.skipnet_batch_size, args.skipnet_learning_rate, args.skipnet_epochs)
	skipnet.predict(args.skipnet_batch_size, args.skipnet_learning_rate)



if __name__ == '__train__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--skipnet_batch_size",
		type=int,
		default=10,
		help="Skipnet batch size")
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