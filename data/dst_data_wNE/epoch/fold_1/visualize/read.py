import numpy as np
# read the npy file and visualize the data
def read_npy(path):
    data = np.load(path)
    return data

label_file = './data/dst_data/epoch/fold_2/val_label2.npy'
read_npy(label_file)