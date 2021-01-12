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


disabled_labels = np.array([1 for _ in range(len(disabled_scans))])
not_disabled_labels = np.array([0 for _ in range(len(not_disabled_scans))])


# Split data in the ratio 70-15-15 for training, validation, and test respectively

x_train = np.concatenate((disabled_scans[21:], not_disabled_scans[89:]), axis=0)
y_train = np.concatenate((disabled_labels[21:], not_disabled_labels[89:]), axis=0)

x_val = np.concatenate((disabled_scans[10:21], not_disabled_scans[44:89]), axis=0)
y_val = np.concatenate((disabled_labels[10:21], not_disabled_labels[44:89]), axis=0)

x_test = np.concatenate((disabled_scans[:10], not_disabled_scans[:44]), axis=0)
y_test = np.concatenate((disabled_labels[:10], not_disabled_labels[:44]), axis=0)


print(
    "Number of samples in train, validation, and test are %d, %d, and %d."
    % (x_train.shape[0], x_val.shape[0], x_test.shape[0])
)

import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)





def get_model(width=64, height=64, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=64, height=64, depth=64)
model.summary()



# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

model_json=model.to_json()
with open("/home/emo4002/colossus_shared3/MSpredict/code/results/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("/home/emo4002/colossus_shared3/MSpredict/code/results/model.h5")
print("saved model to disk")

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

plt.savefig('/home/emo4002/colossus_shared3/MSpredict/code/results/acc_loss_QSM_MSConnect_2mm.png')


# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_test[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "disabled"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that MRI scan is %s"
        % ((100 * score), name)
    )


# Evaluate model.
print("Evaluate on test data")
results=model.evaluate(x_test, y_test)

print("test loss, test acc:", results)
