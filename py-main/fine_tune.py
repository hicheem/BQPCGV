import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from tools import Frame_Level_Aggregation
from tools import Metrics

DATA_FOLDER = 'TRAIN-DATA'
# CSV File contains patches names with dmos score (from parent frame)
#  The csv file contains Patch_Name, DMOS_Score, Parent_Frame_Name
df = pd.read_csv('patches_dmos.csv')

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

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
    horizontal_flip=True
)
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
)


def create_dataset(image_generator, dataframe, shuffle=True):
  dataset = image_generator.flow_from_dataframe(
      dataframe=dataframe,
      directory=DATA_FOLDER,
      x_col='Patch_Name',
      y_col='DMOS',
      target_size=(299, 299),
      batch_size=16,
      class_mode='raw',
      shuffle=shuffle
  )
  return dataset

# df_train, df_test ==> from the global previous df
# fold 4, from cross validation, in my case i used k=4
kfold_train = ['train_fold_1.csv', 'train_fold_2.csv', 'train_fold_3.csv', 'train_fold_4.csv']
kfold_test = ['test_fold_1.csv', 'test_fold_2.csv', 'test_fold_3.csv', 'test_fold_4.csv']
df_train = pd.read_csv(kfold_train[0])
df_test = pd.read_csv(kfold_test[0])
train_data = create_dataset(train_image_generator, df_train)
test_data = create_dataset(test_image_generator, df_test, False)
all_data = create_dataset(train_image_generator, pd.concat([df_train, df_test], ignore_index=True))


def BuildModel(base_model, num_layers_train, dense1, drop1, weights):
  
  # Original unfreeze layers from the previous phase for loading the weights
  num_layers = int(len(base_model.layers) * num_layers_train)
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
  model.load_weights(weights)
  return model

def getModel(cnn):
  
  model = BuildModel(cnn, 0.4, 256, 0.3, 'model_weights.h5')

  # New unfreeze layers for the new phase
  num_layers = int(len(model.layers[0].layers) * 0.1)
  for layer in model.layers[0].layers[-num_layers:]:
    layer.trainable = True
  for layer in model.layers[0].layers[:-num_layers]:
    layer.trainable = False
  return model

model = getModel(cnn)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
# To only train the model on the all dataset
model.fit(all_data, epochs=10)


def fine_tune_and_cross_validation():

    # k-fold-cross-val
    kfold = 4

    # Number of frames available on the dataset
    frames_data = 164
    
    # w => weighted average
    wsrocc = []
    wplcc = []
    wkrcc = []
    wrmse = []
    # avg => average only
    avgsrocc = []
    avgplcc = []
    avgkrcc = []
    avgrmse = []

    for k in range(kfold):

        df_train = pd.read_csv(kfold_train[k])
        df_test = pd.read_csv(kfold_test[k])
        train_data = create_dataset(train_image_generator, df_train)

        model = getModel(cnn)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        model.fit(train_data, epochs=10)

        y_true, y_pred_weight, y_pred_average = Frame_Level_Aggregation(df_test, test_image_generator, create_dataset, model)
        
        srocc, plcc, krcc, rmse = Metrics(y_true, y_pred_weight)
        wsrocc.append(srocc)
        wplcc.append(plcc)
        wkrcc.append(krcc)
        wrmse.append(rmse)
        
        srocc, plcc, krcc, rmse = Metrics(y_true, y_pred_average)
        avgsrocc.append(srocc)
        avgplcc.append(plcc)
        avgkrcc.append(krcc)
        avgrmse.append(rmse)
    
    print(f'Weighted Average || srocc = {np.array(wsrocc).sum()/frames_data} -- plcc = {np.array(wplcc).sum()/frames_data} -- krcc = {np.array(wkrcc).sum()/frames_data} -- rmse = {np.array(wrmse).sum()/frames_data}')
    print(f'Average || srocc = {np.array(avgsrocc).sum()/frames_data} -- plcc = {np.array(avgplcc).sum()/frames_data} -- krcc = {np.array(avgkrcc).sum()/frames_data} -- rmse = {np.array(avgrmse).sum()/frames_data}')
    return wsrocc, wplcc, wkrcc, wrmse, avgsrocc, avgplcc, avgkrcc, avgrmse