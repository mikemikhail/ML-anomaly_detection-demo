#! /usr/bin/env python3

'''
mamikhai@cisco.com
20200111
constructing a modularized version of moniotr-n18h.py
then to be conatinerized for user demoing and exploration at Cisco Live
11/23/2019
monitor-n18g.py
add awareness of weekdays/weekends
10/09/2019
Now this is working great. with normalized features to start from 0 each. with no overlap between features last hour and label/target hour. And with training target (label) being same hour previous day.
Next: will look at:
    1. changing curves to be rate instead of incremental.
    2. checking the best training approach/sequence.
10/03/2019
Will test changing training on 20 minutes previous to a day earlier!

notes from previous release -n18d:
10/01/2019
Previous monitor-n18c.py model is working fine. Here will move to a couple of changes next:
    1. see if model layers, shape can make better prediction of shape, and differentiate shape for each output dimension.
    2. for that be able to reduce the cycle overlap from 50 minutes to like 30m.
10/03/2019
Will put this to work on the server... It is showing tighter fitting to training but a bit wider margin to validation than monitor-n18a.py. Like 114k/138k f
or this versus 129/131 for -n18a. Maybe because of having 20 minute gap versus only 10. Or because of many more hidden nodes, or both. But I like the idea o
f longer gap between training and validation.
'''

import math
# import pandas as pd
# from sklearn import metrics
# from influxdb import DataFrameClient
# import numpy as np
# import tensorflow as tf
# from tensorflow.python.data import Dataset
import time
from datetime import datetime
# from matplotlib import pyplot as plt
import os.path
from constants import *
from read_df import *
from tf_fn import *

# interval = 600 # wait time between prediction cycles, seconds
# hidden_units = [10, 10]
# hidden_units = [72, 36, 18]     # 36, 36 is an overkill!

# set normalization functions to validation (1 hour) ranges
# construct_feature_columns(read_validate(120, 'd_'))

# previous = '1d'
weekday =  datetime.today().weekday()
if weekday == 0:
    previous = '3d'
elif weekday == 5:
    previous = '6d'
else:
    previous = '1d'

if is_demo:
    previous = '1d'

if not os.path.exists(model_directory):
    # first training
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.1,
        learning_rate = 0.1,
        # steps = 3000,
        steps = 3000,
        # batch_size = 120,
        batch_size = 120,
        hidden_units = hidden_units,
        # hidden_units = [80, 80],
        training_examples = read_train_long(2880, 'd_', previous),
        training_targets = read_train_target_long(2880, 'l_', previous),
        validation_examples = read_validate(120, 'd_', previous),
        validation_targets = read_last_target(120, 'v_', previous)
        )
    
    # time.sleep(300)
    time.sleep(interval / 2)

    # second training
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.03,
        learning_rate = 0.03,
        # steps = 3000,
        steps = 3000,
        batch_size = 120,
        hidden_units = hidden_units,
        # hidden_units = [80, 80],
        training_examples = read_train_long(2880, 'd_', previous),
        training_targets = read_train_target_long(2880, 'l_', previous),
        validation_examples = read_validate(120, 'd_', previous),
        validation_targets = read_last_target(120, 'v_', previous)
        )

    # time.sleep(300)
    time.sleep(interval /2)

    # third training
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.001,
        learning_rate = 0.003,
        steps = 3000,
        batch_size = 120,
        hidden_units = hidden_units,
        training_examples = read_train_long(2880, 'd_', previous),
        training_targets = read_train_target_long(2880, 'l_', previous),
        validation_examples = read_validate(120, 'd_', previous),
        validation_targets = read_last_target(120, 'v_', previous)
        )

    # time.sleep(300)
    time.sleep(interval / 2)

cycle = 0

# learn = 0.1
while True:
    weekday =  datetime.today().weekday()
    if weekday == 0:
        previous = '3d'
    elif weekday == 5:
        previous = '6d'
    else:
        previous = '1d'

    if is_demo:
        previous = '1d'

    # if there're pecularities in the 60 minutes data slices, 
    # would be beneficial to retrain on larger dataset, 
    # not necessarily every cycle
    if cycle % 3 == 0: # run every nth cycle
        dnn_regressor = train_nn_regression_model(
            # learning_rate = 0.0003,
            learning_rate = 0.0001,
            steps = 1000,
            batch_size = 120,
            hidden_units = hidden_units,
            training_examples = read_train_long(2880, 'd_', previous, verbose = False),
            training_targets = read_train_target_long(2880, 'l_', previous, verbose = False),
            validation_examples = read_validate(120, 'd_', previous, verbose = False),
            validation_targets = read_last_target(120, 'v_', previous, verbose = False),
            if_plot = False,
            verbose = False
            )

    # train on previous hour set and history, predict on latest hour
    cycle += 1
    print('cycle number ', cycle)
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.0003,
        learning_rate = 0.0003,
        # learning_rate = learn,
        steps = 1000,
        batch_size = 120,
        hidden_units = hidden_units,
        training_examples = read_train(120, 'd_', previous),
        training_targets = read_train_target(120, 'l_', previous),
        validation_examples = read_validate(120, 'd_', previous),
        validation_targets = read_last_target(120, 'v_', previous),
        prediction = True
        )
    # learn /= 5.0
    # time.sleep(600)
    time.sleep(interval)
