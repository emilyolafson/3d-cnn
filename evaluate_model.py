#! /usr/bin/env/python

import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.models import model_from_json

import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 64 # old = 128.
    desired_height = 64
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def normalize(volume):
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


not_disabled_scan_paths = [
    os.path.join(os.getcwd(), "/home/emo4002/colossus_shared3/MSpredict/data/not_disabled", x)
    for x in os.listdir("/home/emo4002/colossus_shared3/MSpredict/data/not_disabled")
]

disabled_scan_paths = [
    os.path.join(os.getcwd(), "/home/emo4002/colossus_shared3/MSpredict/data/disabled", x)
    for x in os.listdir("/home/emo4002/colossus_shared3/MSpredict/data/disabled")
]

print("MRI scans of individuals with not_disabled EDSS: " + str(len(not_disabled_scan_paths)))
print("MRI scans of individuals with disabled EDSS:  " + str(len(disabled_scan_paths)))


# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
disabled_scans = np.array([process_scan(path) for path in disabled_scan_paths])
not_disabled_scans = np.array([process_scan(path) for path in not_disabled_scan_paths])

print("Finished processing scans")

disabled_labels = np.array([1 for _ in range(len(disabled_scans))])
not_disabled_labels = np.array([0 for _ in range(len(not_disabled_scans))])

x_test = np.concatenate((disabled_scans[:10], not_disabled_scans[:44]), axis=0)
y_test = np.concatenate((disabled_labels[:10], not_disabled_labels[:44]), axis=0)

dict={x_test:"x_test", y_test:"y_test"}
pickle_out= open ("/home/emo4002/colossus_shared3/MSpredict/data/test_data.pickle", "wb")
dickle.dump(dict, pickle_out)
pickle_out.close()
print("Successfully saved data to pkl")

resultsdir='/home/emo4002/colossus_shared3/MSpredict/code/results/'
json_file = open(resultsdir+'model.json', 'r')
loaded_model_json = json_file.read()

json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(resultsdir+"model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
loaded_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

print("Evaluate on test data")
results=loaded_model.evaluate(x_test, y_test)

print("test loss, test acc:", results)
