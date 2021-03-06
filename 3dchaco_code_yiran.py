#! /home/emo4002/colossus_shared3/miniconda3/bin/python3
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
from sklearn.model_selection import StratifiedKFold
from scipy import ndimage
import random
from sklearn.utils import class_weight
from keras import backend as K

print("Loading pkled data")

cwd = os.getcwd()
data_dir = str(cwd)+"/data"

pickle_in = open(data_dir + "/all_xdata.pkl", "r+b")
x = pickle.load(pickle_in)

pickle_in = open(data_dir + "/all_ydata.pkl", "r+b")
y = pickle.load(pickle_in)

results_dir=str(cwd) + "/results/6"

@tf.function
def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_model(width=64, height=64, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=10, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=20, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=40, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=40, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="linear")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.AUC(name='auc'),
]


skf=StratifiedKFold(n_splits=10,random_state=7,shuffle=True)
skf_count = 0
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

for train_idx, val_idx in skf.split(x,y):
    print("STARTING MODEL TRAINING FOR SKFOLD SPLIT #: " + str(skf_count + 1))


    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]


    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    batch_size = 2
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
        metrics=[METRICS, "binary_crossentropy"]
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(results_dir + "/model_" + str(skf_count) + ".h5", save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_auc", patience=5)

    # Train the model, doing validation at the end of each epoch
    epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb]
        
    )
    METRICS = [
        keras.metrics.AUC(name='auc'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
  
    ]
    model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[ "binary_crossentropy", METRICS]
)
    # Generate generalization metrics
    scores = model.evaluate(x_val, y_val, verbose=0)
    print(f'Score for fold {skf_count}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
    auc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    model_json=model.to_json()
    with open(results_dir + "/model_" + str(skf_count)+".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(results_dir + "/model_" + str(skf_count) + ".h5")
    print("saved model to disk")

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    plt.savefig(results_dir + "/auc_loss_" + str(skf_count) +".png")
    
    skf_count = skf_count + 1 #increase fold number

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(auc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {auc_per_fold[i]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

