import tensorflow as tf

from .skipnet_functions import *

def preprocess_data():
	features, labels = generate_syn_name_list()
	return train_validation_split(features, labels)

train_features, train_labels, test_features, test_labels = preprocess_data()

def train(batch_size, learning_rate, epochs):
	run_config=tf.estimator.RunConfig(
		model_dir="./models/latest/skipnet_checkpoints",
		save_checkpoints_steps=100,
		save_summary_steps=500)

	estimator = create_estimator(run_config, batch_size, learning_rate)
	train_spec, eval_spec = create_specs(batch_size, epochs, train_features, train_labels, test_features, test_labels)
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



