# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:22:27 2017

데이터 numpy 로 변환
@author: Sarah
"""

import numpy as np
import os
import tensorflow as tf
import glob

os.chdir('C:/Users/KSH/Desktop/yt8m_frame_level')
data_dim = 1024
nb_classes = 4716
len(video_names)



video_names = glob.glob('train*')
for k in range(10):
    print(k, 'th iteration')
    video_lvl_record = video_names[k * 5:((k + 1) * 5)]
    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []

    for j in range(len(video_lvl_record)):
        # print(j)
        for example in tf.python_io.tf_record_iterator(video_lvl_record[j]):

            tf_example = tf.train.Example.FromString(example)
            tmp1 = np.array(tf_example.features.feature['labels'].int64_list.value, dtype='int')
            tmp1.sort()
            if tmp1[0] in range(nb_classes):
                vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
                labels.append(tf_example.features.feature['labels'].int64_list.value)
                mean_rgb.append(tf_example.features.feature['rgb'].float_list.value)
                mean_audio.append(tf_example.features.feature['audio'].float_list.value)

    print('Number of videos in this training set: ', len(mean_rgb))

    y_lab = []
    for j in range(len(labels)):
        a = np.zeros(nb_classes)
        for i in labels[j]:
            if i in range(nb_classes): np.put(a, i, 1)
        y_lab.append(a)

    y_lab = np.array(y_lab, dtype='int8')
    mean_rgb = np.array(mean_rgb, dtype='float32')
    mean_audio = np.array(mean_audio, dtype='float32')
    mean_feature = np.concatenate((mean_rgb, mean_audio), axis=1)
    np.save('D:/IDEA/2017_1_Kaggle_Youtube/video_train/train' + str(k), mean_feature)
    np.save('D:/IDEA/2017_1_Kaggle_Youtube/train_lab/lab' + str(k), y_lab)

# validation set    
video_names = glob.glob('validate*')
for k in range(128):
    print(k, 'th iteration')
    video_lvl_record = video_names[k * 32:((k + 1) * 32)]
    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []

    for j in range(len(video_lvl_record)):
        # print(j)
        for example in tf.python_io.tf_record_iterator(video_lvl_record[j]):
            tf_example = tf.train.Example.FromString(example)
            tmp1 = np.array(tf_example.features.feature['labels'].int64_list.value, dtype='int')
            tmp1.sort()
            if tmp1[0] in range(nb_classes):
                vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
                labels.append(tf_example.features.feature['labels'].int64_list.value)
                mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
                mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)

    print('Number of videos in this training set: ', len(mean_rgb))

    y_lab = []
    for j in range(len(labels)):
        a = np.zeros(nb_classes)
        for i in labels[j]:
            if i in range(nb_classes): np.put(a, i, 1)
        y_lab.append(a)

    y_lab = np.array(y_lab, dtype='int8')
    mean_rgb = np.array(mean_rgb, dtype='float32')
    mean_audio = np.array(mean_audio, dtype='float32')
    mean_feature = np.concatenate((mean_rgb, mean_audio), axis=1)
    np.save('D:/IDEA/2017_1_Kaggle_Youtube/video_validate/validate' + str(k), mean_feature)
    np.save('D:/IDEA/2017_1_Kaggle_Youtube/validate_lab/lab' + str(k), y_lab)

# test set    

vid_ids = []
video_names = glob.glob('train*')
for k=0 in range(10):
    print(k, 'th iteration')
    video_lvl_record = video_names[k * 5:((k + 1) * 5)]
    mean_rgb = []
    mean_audio = []
    for j in range(len(video_lvl_record)):
        # print(j)
        for example in tf.python_io.tf_record_iterator(video_lvl_record[j]):
            tf_example = tf.train.Example.FromString(example)
            vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
            mean_rgb.append(tf_example.features.feature['rgb'].float_list.value)
            mean_audio.append(tf_example.features.feature['audio'].float_list.value)

    print('Number of videos in this training set: ', len(mean_rgb))

    mean_rgb = np.array(mean_rgb, dtype='float32')
    mean_audio = np.array(mean_audio, dtype='float32')
    mean_feature = np.concatenate((mean_rgb, mean_audio), axis=1)
    np.save('C:/Users/KSH/Desktop/yt8m_frame_level/' + str(k), mean_feature)

len(set(vid_ids))
len((vid_ids))
np.save('D:/IDEA/2017_1_Kaggle_Youtube/test_id.npy', vid_ids)
