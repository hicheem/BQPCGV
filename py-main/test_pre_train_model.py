import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm


TEST_DATA_FOLDER = 'TEST-DATA'
TEST_DATA_CSV = 'test_data.csv'
df_test = pd.read_csv(TEST_DATA_CSV)


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


def create_dataset(dataframe, x_col, y_col, directory=None, shuffle=False):
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


test_data = create_dataset(df_test, 'Patch_Name', 'Patch_Score', TEST_DATA_FOLDER)

def BuildModel(base_model, num_layers_to_train, dense1, drop1, weights):
  
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
  model.load_weights(weights)
  return model

# num_layers to unfreeze here is 40% unfreeze layers for loading the weights
model = BuildModel(cnn, 0.4, 256, 0.3, 'model_weights.h5')


from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr,pearsonr, kendalltau
from scipy.optimize import curve_fit

def Calculate_Metrcis():
  y_pred = model.predict(test_data)
  y_true = np.array(df_test['Patch_Score'].values)
  mse = mean_squared_error(y_pred, y_true)
  mae = mean_absolute_error(y_pred, y_true)
  size = len(y_true)

  def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
  beta_init = [np.max(y_true), np.min(y_true), np.mean(y_pred), 0.5]

  y_pred = np.reshape(y_pred,(size)) 
  y_true = np.reshape(y_true,(size))
  popt, _ = curve_fit(logistic_func, y_pred, y_true, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  srocc = spearmanr(y_true,y_pred).correlation
  plcc = pearsonr(y_true,y_pred_logistic)[0]
  rmse = np.sqrt(mse)
  try:
    krcc = kendalltau(y_true, y_pred)[0]
  except:
    krcc = kendalltau(y_true, y_pred, method='asymptotic')[0]
  print(f'srocc = {srocc} -- plcc = {plcc} -- krcc = {krcc} -- rmse = {rmse}')
  return rmse, srocc, plcc, krcc

Calculate_Metrcis()