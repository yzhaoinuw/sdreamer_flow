# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:18:15 2024

@author: yzhao
"""

import os
import numpy as np
from scipy.io import loadmat

from preprocessing import reshape_sleep_data

DATA_PATH = "./data/"
WRITE_PATH = "./data_sdreamer/"
black_list = set([
    "aud_392.mat",
    "aud_403.mat", 
    "chr2_569_yfp.mat", 
    "chr2_573_thres.mat",
    "chr2_580_thres.mat",
    "chr2_590_freq.mat",
    "chr2_590_thres.mat",
    "nor_403_arch.mat",
    "nor_484_yfp.mat",
    "sal_477.mat",
    "sal_602.mat"])

nan_sleep_score_list = []
data_folder = os.listdir(DATA_PATH)
for file in data_folder:
    if not file.endswith(".mat") or file in black_list:
        print(f"skipping {file}")
        continue
    
    print(f"processing {file}.")
    exp_name = file.rstrip(".mat")
    mat_file = DATA_PATH + file
    eeg_reshaped, emg_reshaped, sleep_scores = reshape_sleep_data(mat_file)
    if np.isnan(sleep_scores).any():
        print(f"{file} has nan values in sleep scores.")
        nan_sleep_score_list.append(file)
        continue
        
    sleep_scores = sleep_scores[:, np.newaxis]
    eeg_reshaped = eeg_reshaped[:, np.newaxis, :]
    emg_reshaped = emg_reshaped[:, np.newaxis, :]
    data = np.stack((eeg_reshaped, emg_reshaped), axis=1)
    os.path.join(WRITE_PATH, exp_name, )
    np.save(WRITE_PATH + exp_name + "_data.npy", data) # shape (N, 2, 1, 512)
    np.save(WRITE_PATH + exp_name + "_label.npy", sleep_scores) # shape (N, 2, 1, 512)