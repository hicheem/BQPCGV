import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm


TRAIN_DATA_FOLDER = 'TRAIN-DATA'
TRAIN_DATA_CSV = 'train_data.csv'
df_train = pd.read_csv(TRAIN_DATA_CSV)


base_models = ['densenet121', 'vgg16', 'resnet50', 'xception']

base_model = base_models[0]

if base_model == 'densenet121':
    preprocess_image = tf.keras.applications.densenet
    cnn =  tf.keras.applications.densenet.DenseNet121(
      include_top=False,
      weights='imagenet',
      input_shape=(299, 299, 3),
      pooling="avg"
    )

elif base_model == 'vgg16':
    preprocess_image = tf.keras.applications.vgg16
    cnn =  tf.keras.applications.vgg16.VGG16(
      include_top=False,
      weights='imagenet',
      input_shape=(299, 299, 3),
      pooling="avg"
    )

elif base_model == 'resnet50':
    preprocess_image = tf.keras.applications.resnet50
    cnn =  tf.keras.applications.resnet50.ResNet50(
      include_top=False,
      weights='imagenet',
      input_shape=(299, 299, 3),
      pooling="avg"
    )

elif base_model == 'xception':
    preprocess_image = tf.keras.applications.xception
    cnn =  tf.keras.applications.Xception(
      include_top=False,
      weights='imagenet',
      input_shape=(299, 299, 3),
      pooling="avg"
    )
    
else:
    print('Model is not in: ', base_models)


def preprocess_image(image):
  image = preprocess_image.preprocess_input(image)
  return image

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
)


def create_dataset(dataframe, x_col, y_col, directory=None, shuffle=True):
  dataset = image_generator.flow_from_dataframe(
      dataframe=dataframe,
      directory=directory,
      x_col=x_col,
      y_col=y_col,
      target_size=(299, 299),
      batch_size=32,
      class_mode='raw',
      shuffle=shuffle
  )
  return dataset


train_data = create_dataset(df_train, 'Patch_Name', 'Patch_Score', TRAIN_DATA_FOLDER)

def BuildModel(base_model, num_layers_to_train, dense1, drop1):
  
  # num_layers to unfreeze here % unfreeze layers
  num_layers = int(len(base_model.layers) * num_layers_to_train)
  
  # Iterate over the last % layers and set them as trainable
  for layer in base_model.layers[-num_layers:]:
    layer.trainable = True
  for layer in base_model.layers[:-num_layers]:
    layer.trainable = False
  
  model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(dense1, activation='relu'),
    tf.keras.layers.Dropout(drop1),
    tf.keras.layers.Dense(1, activation='linear')
  ])
  return model


# num_layers to unfreeze here is 40% unfreeze layers
model = BuildModel(cnn, 0.4, 256, 0.3)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_weights.h5',
    monitor='mae',
    mode='min',
    save_best_only=True,
    save_weights_only=True
    )

model.fit(train_data, epochs=20, callbacks=[model_checkpoint_callback])