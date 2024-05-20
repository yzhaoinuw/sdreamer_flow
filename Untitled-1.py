# %%
import torch
import numpy as np

# %%
data_path = '/home/jingyuan_chen/sd/flow/data/dst_data/epoch/'
fold=1
dst_path='{}fold_{}/'.format(data_path, fold)
useNorm = True
train_traces = torch.from_numpy(np.load('{}train_trace{}.npy'.format(dst_path, fold), allow_pickle=True))
train_labels = torch.from_numpy(np.load('{}train_label{}.npy'.format(dst_path, fold), allow_pickle=True))
val_traces = torch.from_numpy(np.load('{}val_trace{}.npy'.format(dst_path, fold), allow_pickle=True))
val_labels = torch.from_numpy(np.load('{}val_label{}.npy'.format(dst_path, fold), allow_pickle=True))
        
train_traces = train_traces[:,:,:1] if not useNorm else train_traces[:,:,-1:]
val_traces = val_traces[:,:,:1] if not useNorm else val_traces[:,:,-1:]

# %%
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef

# %%
X_train = train_traces.reshape(train_traces.shape[0], -1).numpy()
X_val = val_traces.reshape(val_traces.shape[0], -1).numpy()

Y_train = train_labels.reshape(-1).numpy()
Y_val = val_labels.reshape(-1).numpy()


# %%
print('train_traces.shape: ', X_train.shape)
print('train_labels.shape: ', Y_train.shape)

# %%
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1, verbose=True).fit(X_train, Y_train)
svm_predictions = svm_model_linear.predict(X_val)

# %%
accuracy = svm_model_linear.score(X_val, Y_val)
print('accuracy: ', accuracy)
# get f1 score
f1 = f1_score(Y_val, svm_predictions, average='macro')
print('f1: ', f1)
  
# # creating a confusion matrix
# cm = confusion_matrix(Y_val, svm_predictions)


