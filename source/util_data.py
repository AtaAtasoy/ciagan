#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import os
from os import listdir, mkdir
from os.path import isfile, join, isdir, exists
import numpy as np
import importlib
import random
import math
from PIL import Image
from collections import defaultdict
import cv2
import numbers

class ImageDataset(torch.utils.data.Dataset):    
    def __init__(self, root_dir, label_num=1200, transform_fnc=transforms.Compose([transforms.ToTensor()]),
                 img_size = 128, flag_init=True, flag_sample=2, flag_augment=True):

        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        if isinstance(img_size, tuple):
            self.img_shape = img_size
        else:
            self.img_shape = (img_size, img_size)
        self.flag_sample = flag_sample

        self.root_img = 'clr/'
        self.root_lndm = 'lndm/'
        self.root_msk = 'msk/'
        self.im_label, self.im_paths, self.im_index, self.im_names  = [], [], [], []
        
        self.flag_augment = flag_augment
        temp = []
        if flag_init:
            it_j=0
            for it_i in range(label_num):
                folder_list = [f for f in listdir(root_dir+self.root_lndm) if isdir(join(root_dir, self.root_lndm))]
                folder_list.sort()
                                
                imglist_all = [f for f in listdir(join(root_dir, self.root_lndm, folder_list[it_i])) if isfile(join(root_dir, self.root_lndm, folder_list[it_i], f)) and f[-4:] == ".jpg"]
                temp += len(imglist_all) * [folder_list[it_i]]
                imglist_all.sort()

                self.im_label += [it_i] * len(imglist_all)
                self.im_index += [it_j] * len(imglist_all)
                self.im_names += imglist_all
                
                it_j+=1
            temp.sort()
            
            self.im_paths += temp
    def __len__(self):
        return len(self.im_label)

    def load_img(self, im_path):
        im = Image.open(im_path)        
        im = im.resize([int(self.img_shape[0]*1.125)]*2, resample=Image.LANCZOS)
        w, h = im.size

        if self.flag_augment:
            offset_h = 0.
            center_h = h / 2 + offset_h * h
            center_w = w / 2
            min_sz, max_sz = w / 2, (w - center_w) * 1.5
            diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2

            img_res = im.crop(
                (int(center_w - crop_sz - diff_sz * self.crop_rnd[0]), int(center_h - crop_sz - diff_sz * self.crop_rnd[1]),
                 int(center_w + crop_sz + diff_sz * self.crop_rnd[2]), int(center_h + crop_sz + diff_sz * self.crop_rnd[3])))
        else:            
            offset_h = 0.
            center_h = h / 2 + offset_h * h
            center_w = w / 2
            min_sz, max_sz = w / 2, (w - center_w) * 1.5
            crop_sz = self.img_shape[0]/2
            img_res = im.crop(
                (int(center_w - crop_sz),
                 int(center_h - crop_sz),
                 int(center_w + crop_sz),
                 int(center_h + crop_sz)))

        img_res = img_res.resize(self.img_shape, resample=Image.LANCZOS)
        return self.transform_fnc(img_res)

    def __getitem__(self, idx): 
        im_clr, im_lndm, im_msk, im_ind, im_lbl, im_category = [], [], [], [], [], []
        if self.flag_sample==1:
            idx = [idx]
        

        self.im_paths.sort()
        for k_iter in range(self.flag_sample):
            self.crop_rnd = [random.random(), random.random(), random.random(), random.random()]
            im_clr_path = os.path.join(self.root_dir, self.root_img, str(self.im_paths[idx[k_iter]]), self.im_names[idx[k_iter]])
            clr_img = self.load_img(im_clr_path)
            im_clr.append(clr_img)

            im_lndm_path = os.path.join(self.root_dir, self.root_lndm, str(self.im_paths[idx[k_iter]]), self.im_names[idx[k_iter]])
            lndm_img = self.load_img(im_lndm_path)
            im_lndm.append(lndm_img)

            im_msk_path = os.path.join(self.root_dir, self.root_msk, str(self.im_paths[idx[k_iter]]), self.im_names[idx[k_iter]])
            msk = ((1 - self.load_img(im_msk_path)) > 0.2)
            im_msk.append(msk)

            im_ind.append(self.im_index[idx[k_iter]])
            im_lbl.append(self.im_paths[idx[k_iter]])
            im_category.append(self.im_names[idx[k_iter]])
            
        return im_clr, im_lndm, im_msk, im_ind, im_lbl, im_category


def load_data(DATA_PATH, DATA_SET, WORKERS_NUM, BATCH_SIZE, IMG_SIZE, FLAG_DATA_AUGM, LABEL_NUM, mode_train=True):
    ##### Data loaders
    data_dir = DATA_PATH + DATA_SET + '/'
    if mode_train:
        dataset_train = ImageDataset(root_dir=data_dir, label_num=LABEL_NUM, transform_fnc=transforms.Compose([transforms.ToTensor()]),
                                             img_size=IMG_SIZE, flag_augment = FLAG_DATA_AUGM)
        total_steps = int(len(dataset_train) / BATCH_SIZE)

        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_train.im_label):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])
        loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=False, sampler=SiameseSampler(list_of_indices_for_each_class, BATCH_SIZE, total_steps))

        print("Total number of steps per epoch:", total_steps)
        print("Total number of training samples:", len(dataset_train))
        return loader_train, total_steps, LABEL_NUM
    else:
        folder_list = [f for f in listdir(data_dir + 'lndm') if isdir(join(data_dir, 'lndm'))]
        labels = 0
        # Iterate ove each folder and get the number of images
        for folder in folder_list:
            labels += len([f for f in listdir(join(data_dir, 'lndm', folder)) if isfile(join(data_dir, 'lndm', folder, f)) and f[-4:] == ".jpg"])
        print("Total number of labels:", labels)
        
        label_num = labels
        dataset_test = ImageDataset(root_dir=data_dir, label_num=label_num,transform_fnc=transforms.Compose([transforms.ToTensor()]), img_size = IMG_SIZE)
        loader_test = torch.utils.data.DataLoader(dataset=dataset_test, num_workers=1, batch_size=1, shuffle=False)
        print("Total number of test samples:", len(dataset_test))
        return loader_test, len(dataset_test), label_num


class SiameseSampler(Sampler):

    def __init__(self, l_inds, batch_size, iterations_per_epoch):
        self.l_inds = l_inds
        self.max = -1
        self.batch_size = batch_size
        self.flat_list = []
        self.iterations_per_epoch = iterations_per_epoch

    def __iter__(self):
        self.flat_list = []

        for ii in range(int(self.iterations_per_epoch)):
            # get half of the images randomly
            sep = int(self.batch_size / 2)
            for i in range(sep):
                first_class = random.choice(self.l_inds)
                second_class = random.choice(self.l_inds)
                first_element = random.choice(first_class)
                second_element = random.choice(second_class)
                self.flat_list.append([first_element, second_element])

            # get the last half as images from the same class
            for i in range(sep, self.batch_size):
                c_class = random.choice(self.l_inds)
                first_element = random.choice(c_class)
                second_element = random.choice(c_class)
                self.flat_list.append([first_element, second_element])

        random.shuffle(self.flat_list)
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)
