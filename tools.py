from tqdm import tqdm
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr,pearsonr,kendalltau
from scipy.optimize import curve_fit
import numpy as np
import scipy.stats
import scipy.io
import math
from scipy import signal


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat


def Frame_Level_Aggregation(df_data, test_image_generator, create_dataset, model):
  y_true = []
  y_pred_weight = []
  y_pred_average = []

  for frame in tqdm(df_data['Frame_Name'].unique()):
    
    # Get Patches belong to the same frame
    data_frame_patches = df_data['Patch_Name'].map(lambda x: x if frame in x else None)

    data_frame_patches = df_data.loc[data_frame_patches.notnull(), :]
    ds_frame =create_dataset(test_image_generator, data_frame_patches, False)
    
    y_pred = model.predict(ds_frame)
    y_true = data_frame_patches['DMOS'].values
    y_weights = data_frame_patches['Patch_Weight']
    y_pred = np.reshape(y_pred, (len(y_true), ))
    
    nominateur = y_pred * y_weights
    y_weights_sum = tf.reduce_sum(y_weights)
    nominateur_sum = tf.reduce_sum(nominateur)
    frame_score_by_weights = nominateur_sum.numpy() / y_weights_sum.numpy()

    frame_score_by_average = np.mean(y_pred)


    y_true.append(y_true[0])
    y_pred_weight.append(frame_score_by_weights)
    y_pred_average.append(frame_score_by_average)


  y_true = np.array(y_true)
  y_pred_weight = np.array( y_pred_weight)
  y_pred_average = np.array(y_pred_average)
  
  return y_true, y_pred_weight, y_pred_average


# Metrics calculation on Frame Level
def Metrics(y_true, y_pred):
  size = len(y_true)
  beta_init = [np.max(y_true), np.min(y_true), np.mean(y_pred), 0.5]
  y_pred = np.reshape(y_pred,(size)) 
  y_true = np.reshape(y_true,(size))
  popt, _ = curve_fit(logistic_func, y_pred, y_true, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)

  mse = mean_squared_error(y_pred, y_true)
  rmse = np.sqrt(mse)  
  srocc = spearmanr(y_true,y_pred).correlation
  plcc = pearsonr(y_true,y_pred_logistic)[0]
  try:
    krcc = kendalltau(y_true, y_pred)[0]
  except:
    krcc = kendalltau(y_true, y_pred, method='asymptotic')[0]
  return srocc, plcc, krcc, rmse


def hysteresis_pooling(chunk):
    '''parameters'''
    tau = 2 # 2-sec * 30 fps
    comb_alpha = 0.8 # weighting
    ''' function body '''
    chunk = np.asarray(chunk, dtype=np.float64)
    chunk_length = len(chunk)
    l = np.zeros(chunk_length)
    m = np.zeros(chunk_length)
    q = np.zeros(chunk_length)
    for t in range(chunk_length):
        ''' calculate l[t] - the memory component '''
        if t == 0: # corner case
            l[t] = chunk[t]
        else:
            # get previous frame indices
            idx_prev = slice(max(0, t-tau), max(0, t-1)+1)
            # print(idx_prev)
            # calculate min scores 
            l[t] = min(chunk[idx_prev])
        # print("l[t]:", l[t])
        ''' compute m[t] - the current component '''
        if t == chunk_length - 1: # corner case
            m[t] = chunk[t]
        else:
            # get next frame indices
            idx_next = slice(t, min(t + tau, chunk_length))
            # print(idx_next)
            # sort ascend order
            v = np.sort(chunk[idx_next])
            # generated Gaussian weight 
            win_len = len(v) * 2.0 - 1.0
            win_sigma = win_len / 6.0
            # print(win_len, win_sigma)
            gaussian_win = signal.gaussian(win_len, win_sigma)
            gaussian_half_win = gaussian_win[len(v)-1:]
            # normalize gaussian descend kernel
            gaussian_half_win = np.divide(gaussian_half_win, np.sum(gaussian_half_win))
            # print(gaussian_half_win)
            m[t] = sum([x * y for x, y in zip(v, gaussian_half_win)])
        # print("m[t]:", m[t])
    ''' combine l[t] and m[t] into one q[t] '''
    q = comb_alpha * l + (1.0 - comb_alpha) * m
    # print(q)
    # print(np.mean(q))
    return q, np.mean(q)