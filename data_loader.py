import argparse

from src import utils

def main(args):
	utils.load_dataset()
	utils.generate_mask()

if __name__ == '__main__':
	main()