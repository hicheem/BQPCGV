import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats import spearmanr
import numpy as np
import argparse
from tools import hysteresis_pooling


def get_spearman_rankcor(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )

weight_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False,input_shape=(299,299,3),pooling='avg')
densenet =  tf.keras.models.load_model('densenet_v2.h5',custom_objects={'get_spearman_rankcor':get_spearman_rankcor})

def calculate_weights(patches):

  patches = tf.keras.applications.densenet.preprocess_input(patches)
  features = weight_model.predict(patches, verbose=0)
  weights = densenet.predict(features, verbose=0)
  return weights


def image_score(model, image_path, image_name):
    
    print(image_name)
    
    image = Image.open(image_path + image_name).convert('RGB')
    image = np.array(image)

    if image.shape[0] != 1920 or image.shape[1] != 1920:
      image = Image.fromarray(image)
      image = image.resize((1920, 1080), resample=Image.BICUBIC)
      image = np.array(image)
    
    patches = np.zeros((image.shape[0]//299, image.shape[1]//299, 299, 299, 3))                
    for r in range(patches.shape[0]):
      for c in range(patches.shape[1]):
        patches[r,c] = image[r*299:(r+1)*299, c*299:(c+1)*299]
    patches = patches.reshape((-1, 299, 299, 3))
    
    # Predict Scores Patches
    preds = model.predict(tf.keras.applications.densenet.preprocess_input(patches), verbose = 0)
    # Predict Weights Patches
    weights = calculate_weights(patches)

    # Image Aggregation using Weights
    nominateur = preds * weights
    weights_sum = tf.reduce_sum(weights)
    nominateur_sum = tf.reduce_sum(nominateur)
    
    # Final Image Score by Weighted Average
    image_score_weight = nominateur_sum.numpy() / weights_sum.numpy()
   
    # Final Image Score by Average
    image_score_average = np.mean(preds)

    scores = {
       "mos_w" : image_score_weight,
       "mos_avg" : image_score_average,
    }
    return scores


def video_pooling(model, video_path, video_name, number_frames_per_second):

  print(video_name)
  video = cv2.VideoCapture(video_path + video_name)
  # Get video properties
  fps = video.get(cv2.CAP_PROP_FPS)
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  
  
  frame_interval = int(fps / number_frames_per_second)

  
  frames_scores_by_average = []
  frames_scores_by_weights = []
  MOS_by_patches_weights_frame_weight = []
  MOS_by_patches_average_frame_weight = []
  all_frames_weights_sum = 0
  
  MOS_AVERAGE_Patches_Average_Frames = []
  MOS_WEIGHT_Patches_Frames_Average = []
  
  bar_plot = tqdm(range(0, num_frames, frame_interval))
  for f in bar_plot:

    #read custom frame per second -------
    video.set(cv2.CAP_PROP_POS_FRAMES, f)
    #-----------------------
    ret, frame = video.read()
    
    # Check if we have reached the end of the video
    if not ret:
      break

    if width != 1920:
      frame = Image.fromarray(frame)
      frame = frame.resize((1920, 1080), resample=Image.BICUBIC)
      frame = np.array(frame)
      
    
    # Extracted patches from the current frame
    patches = np.zeros((frame.shape[0]//299, frame.shape[1]//299, 299, 299, 3))                
    for r in range(patches.shape[0]):
      for c in range(patches.shape[1]):
        patches[r,c] = frame[r*299:(r+1)*299, c*299:(c+1)*299]
    patches = patches.reshape((-1, 299, 299, 3))
    
    # Predict Scores Patches
    preds = model.predict(tf.keras.applications.densenet.preprocess_input(patches), verbose = 0)
    # Predict Weights Patches
    weights = calculate_weights(patches)

    # Frame Aggregation using Weights
    nominateur = preds * weights
    weights_sum = tf.reduce_sum(weights)
    nominateur_sum = tf.reduce_sum(nominateur)
    # Final Frame Score by Weighted Average
    frame_score_weight = nominateur_sum.numpy() / weights_sum.numpy()
    
    # Convert the PIL Image to numpy array from getting its weight
    frame = Image.fromarray(frame)
    frame = frame.resize((299, 299), resample=Image.BICUBIC)
    frame = np.expand_dims(np.array(frame), axis=0)
    frame_weight = calculate_weights(frame)[0]


    # Final Frame Score by Average
    frame_score_avg = np.mean(preds)
    
    
    # Appending each frame score, for using pooling after.

    MOS_AVERAGE_Patches_Average_Frames.append(frame_score_avg)
    MOS_WEIGHT_Patches_Frames_Average.append(frame_score_weight)
    
    frames_scores_by_average.append(frame_score_avg)
    frames_scores_by_weights.append(frame_score_weight)
    
    MOS_by_patches_weights_frame_weight.append(frame_score_weight * frame_weight[0])
    MOS_by_patches_average_frame_weight.append(frame_score_avg * frame_weight[0])
    all_frames_weights_sum += frame_weight[0]

  
    del frame_score_avg, preds, patches
    bar_plot.set_postfix({'frame': f'{f}/{num_frames}'})
  
  
  # Get Final Video Score for Each method used 

  MOS_WEIGHT_Patches_Frames_Average = np.mean(MOS_WEIGHT_Patches_Frames_Average)
  MOS_AVERAGE_Patches_Average_Frames = np.mean(MOS_AVERAGE_Patches_Average_Frames)

  MOS_by_patches_weights_frame_weight = np.array(MOS_by_patches_weights_frame_weight).sum() / all_frames_weights_sum
  MOS_by_patches_average_frame_weight = np.array(MOS_by_patches_average_frame_weight).sum() / all_frames_weights_sum

  _, MOS_by_Patches_Weights_Temp_Hys = hysteresis_pooling(frames_scores_by_weights)
  _, MOS_by_Patches_Average_Temp_Hys = hysteresis_pooling(frames_scores_by_average)

  # Close the Video
  video.release()
  
  scores = {
    # mos by weighted average patches, average video
    "mos_w_avg": MOS_WEIGHT_Patches_Frames_Average,
    
    # mos by average patches, average video
    "mos_avg_avg": MOS_AVERAGE_Patches_Average_Frames,

    # mos by weighted average patches, weighted average frames
    "mos_w_w": MOS_by_patches_weights_frame_weight,
    
    # mos by average patches, weighted average frames
    "mos_avg_w": MOS_by_patches_average_frame_weight,

    # mos by weighted average patches, Temporal Hysteresis video
    "mos_w_hys": MOS_by_Patches_Weights_Temp_Hys,
    
    # mos by average patches, Temporal Hysterisis video
    "mos_avg_hys": MOS_by_Patches_Average_Temp_Hys

  }

  return scores

def test_image(model, imagepath, imagename):
    BQPGV = tf.keras.models.load_model(model)
    scores = image_score(BQPGV, imagepath, imagename)

    print("MOS based on Weighted Average Aggregation:",scores['mos_w'])
    print("MOS based on Average Aggregation:",scores['mos_avg'])

    csv_file = imagename.split('.')[0] + '_predicted_dmos.csv'
    header = ['Image Name', 'mos_w', 'mos_avg']
    with open(csv_file, 'a') as w:
        w.write(','.join(header) + '\n')
        data = ','.join([imagename, str(scores['mos_w']), str(scores['mos_avg']) ])+ '\n'
        w.write(data)

def test_video(model, videopath, videoname, number_frames_per_second):

    BQPGV = tf.keras.models.load_model(model)
    scores = video_pooling(BQPGV, videopath, videoname, number_frames_per_second)


    print("MOS based on Weighted Average Frame Level & Average Pooling Video Level:",scores['mos_w_avg'])
    print("MOS based on Average Frame Level & Average Pooling Video Level:",scores['mos_avg_avg'])
    print("MOS based on Weighted Average Frame Level & Weighted Average Pooling Video Level:",scores['mos_w_w'])
    print("MOS based on Average Frame Level & Weighted Average Pooling Video Level:",scores['mos_avg_w'])
    print("MOS based on Weighted Average Frame Level & Hysteresis Pooling Video Level:",scores['mos_w_hys'])
    print("MOS based on Average Frame Level & Hysteresis Pooling Video Level:",scores['mos_avg_hys'])

    
    csv_file = videoname.split('.')[0] + '_predicted_dmos.csv'
    header = ['Video Name', 'mos_w_avg', 'mos_avg_avg', 'mos_w_w', 'mos_avg_w', 'mos_w_hys', 'mos_avg_hys']
    with open(csv_file, 'a') as w:
        w.write(','.join(header) + '\n')
        data = ','.join([videoname, str(scores['mos_w_avg']), str(scores['mos_avg_avg']), str(scores['mos_w_w']), str(scores['mos_avg_w']), str(scores['mos_w_hys']), str(scores['mos_avg_hys']) ])+ '\n'
        w.write(data)


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', action='store', dest='model', default=r'./models/model_Final_DMOS.h5' ,
                    help='Specify the model path , e.g. ./models/model_Final_DMOS.h5')
                    
    parser.add_argument('-vp', '--videopath', action='store', dest='videopath', default=r'./videos/' ,
                    help='Specify the folder video path, e.g. ./videos/ ')
                    
    parser.add_argument('-vn', '--videoname', action='store', dest='videoname', default='video1.mp4' ,
                    help='Specify the folder the video name, e.g. sample.mp4')
    parser.add_argument('-fps', '--framepersecond', action='store', dest='number_frames_per_second', default='1' ,
                    help='Specify the selected frames_per_second , e.g. 1')
    
    
    values = parser.parse_args()

    test_video(values.model, values.videopath, values.videoname, int(values.number_frames_per_second))
        