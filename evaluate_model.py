#! /usr/bin/env/python

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pickle
import nibabel as nib

from scipy import ndimage


cwd = os.getcwd()
data_dir = str(cwd)+"/data"
pickle_in = open(data_dir + "/xtest_data.pkl", "r+b")
x_test = pickle.load(pickle_in)

pickle_in = open(data_dir + "/ytest_data.pkl", "r+b")
y_test = pickle.load(pickle_in)

pickle_in = open(data_dir + "/xtrain_data.pkl", "r+b")
x_train = pickle.load(pickle_in)

pickle_in = open(data_dir + "/ytrain_data.pkl", "r+b")
y_train = pickle.load(pickle_in)

pickle_in = open(data_dir + "/xval_data.pkl", "r+b")
x_val = pickle.load(pickle_in)

pickle_in = open(data_dir + "/yval_data.pkl", "r+b")
y_val = pickle.load(pickle_in)

print(
    "Number of samples in train, validation, and test are %d, %d, and %d."
    % (x_train.shape[0], x_val.shape[0], x_test.shape[0])
)


resultsdir='/home/emo4002/colossus_shared3/MSpredict/3d-cnn/'
json_file = open(resultsdir+'model.json', 'r')
loaded_model_json = json_file.read()

json_file.close()
loaded_model = model_from_json(loaded_model_json)


# evaluate loaded model on test data
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

loaded_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[METRICS, "binary_crossentropy"]
)


print("Evaluate on test data")
results=loaded_model.evaluate(x_test, y_test)

print("test loss, test acc:", results)
