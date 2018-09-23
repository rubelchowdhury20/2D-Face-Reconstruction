import tensorflow as tf
import numpy as np
from sfsnet_arch import *
from sfsnet_functions import *

def preprocess_data(batch_size):

    features, labels = generate_list()
    train_data, test_data = train_validation_split(features, labels, batch_size)

    return train_data, test_data

def train(batch_size, learning_rate, epochs):
    train_data, test_data = preprocess_data(batch_size)
    estimator, train_spec, eval_spec = create_estimator_and_specs(
      train_data,
      test_data,
      batch_size,
      learning_rate,
      epochs,
      run_config=tf.estimator.RunConfig(
        model_dir="./models/latest/sfsnet_checkpoints",
        save_checkpoints_steps=100,
        save_summary_steps=500))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)    
