import argparse

from src import utils

def main():
	utils.load_dataset()
	utils.generate_mask()

if __name__ == '__main__':
	main()