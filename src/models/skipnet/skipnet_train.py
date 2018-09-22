import tensorflow as tf

from .skipnet_arch import *
from .skipnet_functions import *

def preprocess_data():
	features, labels = generate_syn_name_list()
	return train_validation_split(features, labels)

train_features, train_labels, test_features, test_labels = preprocess_data()

def train(batch_size, learning_rate, epochs):
	estimator = create_estimator(batch_size, learning_rate)
	train_spec, eval_spec = create_specs(batch_size, epochs, train_features, train_labels, test_features, test_labels)
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



