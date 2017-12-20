from sklearn.preprocessing import StandardScaler
from sklearn import datasets, model_selection
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops as rr
import numpy as np; np.seed=5
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from tqdm import tqdm
import pandas as pd

from custom_op import *


def build_network(n_input, 
                  n_output, 
                  n_hidden, 
                  n_layers,
                  ini_mean,
                  ini_stddev,
                  output_type,
                  
                  dgd_window=1
                 ):

    # the network

    hidden_dict = {}

    num_params = (n_input*n_hidden) + n_hidden + (n_hidden*n_hidden*(n_layers)) + (n_output*n_hidden) + n_output


    activation = tf.nn.relu

    def weight_matrix(n_in,n_out):
        return tf.Variable(
            tf.truncated_normal(shape=(n_in,n_out), mean=ini_mean, stddev=ini_stddev)
            ,dtype=tf.float32)

    def bias_matrix(n_to):
        return tf.Variable(tf.zeros(shape=(1,n_to),dtype=tf.float32))



    input_layer = tf.placeholder(shape=(None,n_input),dtype=tf.float32)
    output_truth = tf.placeholder(shape=(None,n_output),dtype=tf.float32)
    gamma = tf.placeholder(shape=(),dtype=tf.float32)


    for n in range(1,n_layers+1):
        hidden_dict[n] = {}
        if n == 1:
            hidden_dict[n]['weights'] = weight_matrix(n_input,n_hidden)
            hidden_dict[n]['bias'] = bias_matrix(n_hidden)
            hidden_dict[n]['layer'] = activation(tf.matmul(input_layer,hidden_dict[n]['weights']) + hidden_dict[n]['bias'])
        else:
            hidden_dict[n]['weights'] = weight_matrix(n_hidden,n_hidden)
            hidden_dict[n]['bias'] = bias_matrix(n_hidden)
            hidden_dict[n]['layer'] = activation(tf.matmul(hidden_dict[n-1]['layer'],hidden_dict[n]['weights']) + hidden_dict[n]['bias'])


    output_weights = weight_matrix(n_hidden,n_output)
    output_bias = bias_matrix(n_output)

    output_pre = tf.matmul(hidden_dict[n_layers]['layer'],output_weights) + output_bias
    
    if output_type == 'sigmoid':
        output_pred = tf.nn.sigmoid(output_pre)
    elif output_type == 'linear':
        output_pred = output_pre

    loss = tf.reduce_sum(tf.reduce_mean((1/2)*(output_truth - output_pred)**2))
    #loss = -tf.reduce_sum(tf.reduce_mean(output_truth*tf.log(output_pred)+(1-output_truth)*tf.log(1-output_pred)))

    lr = tf.placeholder(shape=(),dtype=tf.float32)
    #opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False)
    #opt = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)

    all_variables = [hidden_dict[n]['weights'] for n in range(1,n_layers+1)] + [output_weights] + [hidden_dict[n]['bias'] for n in range(1,n_layers+1)] + [output_bias]
    processors = [GradientProcessor(shape = variable.shape, gamma = gamma, window = dgd_window) for variable in all_variables]

    control_opt = opt.minimize(loss, var_list = all_variables)

    #correct_prediction = tf.equal(output_truth, tf.round(output_pred))
    #acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    grads_and_vars = opt.compute_gradients(loss, all_variables)

    processed_gradients = process_gradients(processors, grads_and_vars)
    with tf.control_dependencies([g[0] for g in processed_gradients]):
        custom_opt = opt.apply_gradients(processed_gradients)

    print('network built')
    
    return input_layer, output_truth, lr, gamma, loss, control_opt, custom_opt,  output_pred, grads_and_vars, processed_gradients


def main():
    
    print('starting')
    
    input_layer, output_truth, lr, gamma, loss, control_opt, custom_opt,  output_pred, grads_and_vars, processed_gradients = build_network(n_input = 13, 
                                                          n_output = 1, 
                                                          n_hidden = 100, 
                                                          n_layers = 20,
                                                          ini_mean = 0,
                                                          ini_stddev = 1/(2*100),
                                                          output_type = 'linear',

                                                          dgd_window=1
                                                         )
    
if __name__ == '__main__':
    main()
    
    