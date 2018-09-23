import tensorflow as tf
import numpy as np
from sfsnet_arch import *
from sfsnet_functions import *

def preprocess_data():

    features, labels = generate_list()
    train_data, test_data = train_validation_split(features, labels)

    return train_data, test_data

def train(batch_size, learning_rate, epochs):

    train_data, test_data = preprocess_data()
    estimator, train_spec, eval_spec = create_estimator_and_specs(run_config=tf.estimator.RunConfig(
                                                              model_dir="./checkpoints/sfsnet_checkpoints",
                                                              save_checkpoints_steps=100,
                                                              save_summary_steps=500), train_data, test_data,
                                                              batch_size, learning_rate, epochs)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)    
