#! /usr/bin/env python3

'''
mamikhai@cisco.com
20200111
Tensorflow fnd plot unctions for monitor.py
'''

import math
from sklearn import metrics
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from matplotlib import pyplot as plt
from constants import *

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  global feature_mean
  global feature_std
  global feature_max
  if feature_mean == 0:
    feature_mean = input_features.mean().mean()
    print('feature mean: ', feature_mean)
  if feature_std == 0:
    feature_std = input_features.std().mean()
    print('feature std: ', feature_std)
  if feature_max == 0:
    feature_max = input_features.max().mean() / 24 # The mean max per 1 hour
    print('feature max: ', feature_max)

  # epsilon = 0.000001
  epsilon = 0.0

  # choose best normalization of input data
  '''
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - feature_mean) / (feature_std))
              for my_feature in input_features])
  
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - input_features[my_feature].mean()) / (input_features[my_feature].std()))
              for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val) / (input_features[my_feature].max()))
              for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - input_features[my_feature].mean()) / (input_features[my_feature].max()))
              for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - int(input_features.mean()[my_feature])) / (int(input_features.std()[my_feature])))
              for my_feature in input_features])
  '''
  # return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - feature_mean) / (feature_std))
  #             for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val) / (feature_max))
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
    """Trains a neural net regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                             
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_regression_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    if_plot = True,
    prediction = False,
    verbose = True):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing one or more columns 
      to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column
      to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns
      to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column
      to use as target for validation.
      
  Returns:
    A `DNNRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # for Tensorflow 1.x
  # my_optimizer = tf.optimizers.Adam(learning_rate=learning_rate) # for Tensorflow 2.x
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0) # for Tensorflow 1.x
  # my_optimizer = tf.clip_by_norm(my_optimizer, 5.0) # for Tensorflow 2.x
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer,
      model_dir= model_directory,
      label_dimension= len(tunnel_ifs) + len(physical_ifs)
  )
  
  # Create input functions.
  # print(training_targets)
  # print(training_examples)
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets, 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets, 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    # training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    # training_predictions = np.array([[item['predictions'][0], item['predictions'][1], item['predictions'][2], item['predictions'][3], item['predictions'][4], item['predictions'][5], item['predictions'][6], item['predictions'][7], item['predictions'][8], item['predictions'][9]] for item in training_predictions])
    training_predictions = np.array([[item['predictions'][i] for i in range(0, len(tunnel_ifs) + len(physical_ifs))] for item in training_predictions])

    if verbose:
        print('training_predictions baoundaries')
        print(training_predictions[0])
        print(training_predictions[len(training_predictions)-1])

    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    # validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    # validation_predictions = np.array([[item['predictions'][0], item['predictions'][1], item['predictions'][2], item['predictions'][3], item['predictions'][4], item['predictions'][5], item['predictions'][6], item['predictions'][7], item['predictions'][8], item['predictions'][9]] for item in validation_predictions])
    validation_predictions = np.array([[item['predictions'][i]for i in range(0, len(tunnel_ifs) + len(physical_ifs))] for item in validation_predictions])

    if verbose:
        print('validation_predictions boundaries')
        print(validation_predictions[0])
        print(validation_predictions[len(validation_predictions)-1])
    
    '''
    # validation plot: if you want to plot every period (slow)
    if if_plot:
        fig = plt.figure(2, figsize=[14, 7])
        fig.clear()
        fig.suptitle('bytes-sent count vs. records', fontsize=16)
        fig_rows = 3
        fig_cols = int((len(tunnel_ifs) + len(physical_ifs)) / fig_rows)
        ifs_plots = fig.subplots(fig_rows, fig_cols, sharex='all', gridspec_kw={'hspace':0.1, 'wspace':0.3, 'left':0.04, 'right':0.99, 'top':0.93, 'bottom':0.03})
        for dim in range(0, len(tunnel_ifs) + len(physical_ifs)):
            ifs_plot = ifs_plots[int(dim / fig_cols), dim % fig_cols]
            ifs_plot.plot(validation_targets.values[:,dim], 'g')
            ifs_plot.plot(validation_predictions[:,dim], 'b')
            ifs_plot.grid(True, which='both', axis='y')
        plt.xticks([0, 59, 119])
        plt.figlegend(["actual", "prediction"], loc='upper right')
        plt.pause(0.0001)
    '''

    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  if if_plot:
    # RMSE values and graphs
    global x_periods
    x_periods += periods
    plt.ion()

    validation_nrmse_prediction = validation_root_mean_squared_error / validation_predictions.mean()
    validation_nrmse_actual = validation_root_mean_squared_error / validation_targets.mean().mean()
  
    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
    print("Final NRMSE (/prediction, /actual): %0.2f, %0.2f" % (validation_nrmse_prediction, validation_nrmse_actual))
    # print(validation_predictions.mean())
    # print(validation_targets.max().mean())
    # validation plot for each dimension
    fig = plt.figure(1, figsize=[14, 7])
    fig.clear()
    fig.suptitle('bytes-sent count vs. records', fontsize=16)
    fig_rows = 3
    fig_cols = int((len(tunnel_ifs) + len(physical_ifs)) / fig_rows)
    ifs_plots = fig.subplots(fig_rows, fig_cols, sharex='all', gridspec_kw={'hspace':0.1, 'wspace':0.3, 'left':0.04, 'right':0.99, 'top':0.93, 'bottom':0.03})
    # plt.ylabel("bytes-sent")
    # plt.xlabel("records")
    for dim in range(0, len(tunnel_ifs) + len(physical_ifs)):
        ifs_plot = ifs_plots[int(dim / fig_cols), dim % fig_cols]
        ifs_plot.plot(validation_targets.values[:,dim], 'g')
        ifs_plot.plot(validation_predictions[:,dim], 'b')
        ifs_plot.grid(True, which='both', axis='y')
        ifs_plot.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
    plt.xticks([0, 59, 119])
    plt.figlegend(["actual", "prediction"], loc='upper right')
    plt.pause(0.0001)

    # graph of loss metrics over periods
    plt.figure(2, figsize=[10, 4])
    plt.ylabel("RMSE (log scale)")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.yscale('log')
    plt.tight_layout()
    plt.plot(range(x_periods - 10, x_periods), training_rmse, 'm')
    plt.plot(range(x_periods - 10, x_periods), validation_rmse, 'r')
    plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1], 'b*')
    plt.legend(["training", "validation", "prediction"])
    if not prediction:
        plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1], 'r*')
    plt.grid(True, which='both', axis='both')
    plt.pause(0.0001)
 
    # graph of normalized loss metrics over periods
    plt.figure(3, figsize=[10, 4])
    plt.ylabel("NRMSE")
    plt.xlabel("Periods")
    plt.title("Prediction RMSE/mean vs. Periods")
    # plt.yscale()
    plt.tight_layout()
    plt.plot(range(x_periods - 10, x_periods), validation_rmse / validation_predictions.mean(), 'b')
    plt.plot(range(x_periods - 10, x_periods), validation_rmse / validation_targets.mean().mean(), 'g')
    plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1] / validation_predictions.mean(), 'b*')
    plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1] / validation_targets.mean().mean(), 'g*')
    plt.legend(["/prediction", "/actual"])
    plt.grid(True, which='both', axis='both')
    plt.pause(0.0001)

  return dnn_regressor

