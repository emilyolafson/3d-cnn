
import os
import zipfile
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.models import model_from_json
import nibabel as nib
from scipy import ndimage


cwd = os.getcwd()
data_dir = str(cwd)+"/data"

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
    os.path.join(os.getcwd(), data_dir + "/not_disabled", x)
    for x in os.listdir(data_dir + "/not_disabled")
]

disabled_scan_paths = [
    os.path.join(os.getcwd(),data_dir + "/disabled", x)
    for x in os.listdir(data_dir + "/disabled")
]

print("MRI scans of individuals with not_disabled EDSS: " + str(len(not_disabled_scan_paths)))
print("MRI scans of individuals with disabled EDSS:  " + str(len(disabled_scan_paths)))

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
disabled_scans = np.array([process_scan(path) for path in disabled_scan_paths])
not_disabled_scans = np.array([process_scan(path) for path in not_disabled_scan_paths])

disabled_labels = np.array([1 for _ in range(len(disabled_scans))])
not_disabled_labels = np.array([0 for _ in range(len(not_disabled_scans))])

# Split data in the ratio 70-15-15 for training, validation, and test respectively

x_train = np.concatenate((disabled_scans[21:], not_disabled_scans[89:]), axis=0)
y_train = np.concatenate((disabled_labels[21:], not_disabled_labels[89:]), axis=0)

x_val = np.concatenate((disabled_scans[10:21], not_disabled_scans[44:89]), axis=0)
y_val = np.concatenate((disabled_labels[10:21], not_disabled_labels[44:89]), axis=0)

x_test = np.concatenate((disabled_scans[:10], not_disabled_scans[:44]), axis=0)
y_test = np.concatenate((disabled_labels[:10], not_disabled_labels[:44]), axis=0)

pickle_out= open(data_dir + "/xtest_data.pickle", "wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out= open(data_dir + "ytest_data.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_out= open(data_dir + "/xtrain_data.pickle", "wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out= open(data_dir + "/ytrain_data.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out= open(data_dir + "/xval_data.pickle", "wb")
pickle.dump(x_val, pickle_out)
pickle_out.close()

pickle_out= open(data_dir + "/yval_data.pickle", "wb")
pickle.dump(y_val, pickle_out)
pickle_out.close()

print("Saved data to pkl format")