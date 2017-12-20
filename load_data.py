import numpy as np; np.seed=5
from sklearn import datasets, model_selection


def load_saved(dir_name):
    inputs_train = np.load('{}/inputs_train.npy'.format(dir_name))
    inputs_test = np.load('{}/inputs_test.npy'.format(dir_name))
    outputs_train = np.load('{}/outputs_train.npy'.format(dir_name))
    outputs_test = np.load('{}/outputs_test.npy'.format(dir_name))
    
    return inputs_train, inputs_test, outputs_train, outputs_test

def load_new(dataset_name = 'boston'):
    
    if dataset_name == 'boston':
        data = datasets.load_boston()
    elif dataset_name == 'breast cancer':
        data = datasets.load_breast_cancer()
  
    indata = data.data
    outdata = data.target.reshape(-1,1)
    in_scaler = StandardScaler()
    indata = in_scaler.fit_transform(indata)
    inputs_train, inputs_test, outputs_train, outputs_test = model_selection.train_test_split(indata, outdata, test_size=0.1)
    
    return inputs_train, inputs_test, outputs_train, outputs_test