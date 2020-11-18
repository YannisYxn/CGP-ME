# -*- coding: utf-8 -*-
# Author: YeXiaona
# Date  : 2020-11-11
# data_loader

import os
import time

import cv2
import numpy as np

def load_data():
    print("=================data loading=================")
    prefix_path = os.getcwd()
    data_path = prefix_path + '/data/flow0/'
    emotion = ['ne', 'po', 'sur']

    train_data_list = []
    test_data_list = []
    start = time.process_time()

    for file in os.listdir(data_path):
        if file == 'total': continue
        train_X, validate_X, test_X, train_label, validate_label, test_label = [], [], [], [], [], []

        train_path = data_path + file + '/train/'

        for pic in os.listdir(train_path):
            if pic.split('_')[-1] == 'os.tif':
                img = cv2.imread(train_path + pic)
                img = cv2.resize(img, (28, 28))
                subimg_u = cv2.imread(train_path + "_".join(pic.split('_')[:-1]) + "_u.tif")
                subimg_u = cv2.resize(subimg_u, (28, 28))
                subimg_v = cv2.imread(train_path + "_".join(pic.split('_')[:-1]) + "_v.tif")
                subimg_v = cv2.resize(subimg_v, (28, 28))
                subimg_vis = cv2.imread(train_path + "_".join(pic.split('_')[:-1]) + "_vis.tif")
                subimg_vis = cv2.resize(subimg_vis, (28, 28))

                train_X.append(np.array(img).flatten().tolist() +
                               np.array(subimg_u).flatten().tolist() +
                               np.array(subimg_v).flatten().tolist() +
                               np.array(subimg_vis).flatten().tolist() )
                label = [i for i, each in enumerate(emotion) if each in pic]
                train_label.append(label[0])

        test_path = data_path + file + '/test/'
        for pic in os.listdir(test_path):
            if pic.split('_')[-1] == 'os.tif':
                img = cv2.imread(test_path + pic)
                img = cv2.resize(img, (28, 28))
                subimg_u = cv2.imread(test_path + "_".join(pic.split('_')[:-1]) + "_u.tif")
                subimg_u = cv2.resize(subimg_u, (28, 28))
                subimg_v = cv2.imread(test_path + "_".join(pic.split('_')[:-1]) + "_v.tif")
                subimg_v = cv2.resize(subimg_v, (28, 28))
                subimg_vis = cv2.imread(test_path + "_".join(pic.split('_')[:-1]) + "_vis.tif")
                subimg_vis = cv2.resize(subimg_vis, (28, 28))

                test_X.append(np.array(img).flatten().tolist() +
                               np.array(subimg_u).flatten().tolist() +
                               np.array(subimg_v).flatten().tolist() +
                               np.array(subimg_vis).flatten().tolist())
                label = [i for i, each in enumerate(emotion) if each in pic]
                test_label.append(label[0])

        train_data_list.append(np.hstack((train_X, np.array(train_label).reshape(len(train_label),1))))
        test_data_list.append(np.hstack((test_X, np.array(test_label).reshape(len(test_label),1))))

    end = time.process_time()
    print("=============data loaded in {} s =============".format(int(end - start)))

    return np.array(train_data_list), np.array(test_data_list)