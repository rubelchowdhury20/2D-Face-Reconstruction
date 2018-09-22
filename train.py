import argparses

from src import utils
from src.models import skipnet
from src.models import sfsnet


def main(args):
	skipnet.train(args.skipnet_batch_size, args.skipnet_learning_rate, args.skipnet_epochs)
	skipnet.predict(args.skipnet_batch_size, args.skipnet_learning_rate)
	#sfsnet.train(args.sfsnet_batch_size, args.sfsnet_learning_rate, args.sfsnet_epochs)
	#sfsnet.predict(args.sfsnet_batch_size, args.sfsnet_learning_rate) # first argument is test data

if __name__ == '__main__':
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
	parser.add_argument(
		"--sfsnet_batch_size",
		type=int,
		default=10,
		help="sfsnet batch size")
	parser.add_argument(
		"--sfsnet_learning_rate",
		type=float,
		default=0.00001,
		help="sfsnet learning rate")
	parser.add_argument(
		"--sfsnet_epochs",
		type=int,
		default=10,
		help="sfsnet total epochs")

	main(parser.parse_args())