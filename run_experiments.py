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
import time

from build_network import *
from load_data import *


def run_training(inputs, 
                 outputs, 
                 network_nodes,
                 n_epochs, 
                 learning_rate, 
                 _gamma, 
                 tqdm_on=False, 
                 batch_size=64, 
                 control=False,
                 
                ):
    
    inputs_train, inputs_test = inputs
    outputs_train, outputs_test = outputs
    
    if tqdm_on is True: pbar = tqdm(range(n_epochs))
    else: pbar = range(n_epochs)
    train_losses = []
    test_losses = []
    grads = []
    proc_grads = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        if control is True:
            optimizer = network_nodes['control_opt']
        else:
            optimizer = network_nodes['custom_opt']
        
        for _ in pbar:
            
            batch_idx = np.random.choice(np.arange(len(inputs_train)), size = batch_size)
            input_batch = inputs_train[batch_idx]
            output_batch = outputs_train[batch_idx]
            
            
            
            _, _, _loss, pred, _  = sess.run([optimizer, 
                                              network_nodes['processed_gradients'], 
                                              network_nodes['loss'], 
                                              network_nodes['output_pred'], 
                                              network_nodes['grads_and_vars']], 
                                                  feed_dict={network_nodes['input_layer']: input_batch
                                                             , network_nodes['output_truth']: output_batch
                                                             , network_nodes['lr']: learning_rate
                                                             , network_nodes['gamma']: _gamma
                                                            })
            
            _test_loss = sess.run(network_nodes['loss'], feed_dict={network_nodes['input_layer']:inputs_test, network_nodes['output_truth']:outputs_test})
            
            train_losses.append(_loss)
            test_losses.append(_test_loss)
            #grads.append(_grads)
            #proc_grads.append(_proc_grads)
            
            if tqdm_on is True:
                pbar.set_description('train: {:2f}, test: {:2f}'.format(_loss,_test_loss))

    return train_losses, test_losses, grads, proc_grads





def run_exps(num_exps, 
             network_params,
             network_nodes,
             tqdm_on=False, 
             epochs=15, 
             lrate=0.1, 
             _gamma=0.5, 
             batch_size=64, 
             control=False,
             
             inputs_train = [], 
             inputs_test = [],
             outputs_train = [], 
             outputs_test = [],
            ):
    
    df_test = None
    times = []
    
    name = 'l{}_h{}_bs{}_lr{}_g{}_w{}_adam'.format(network_params['n_layers'], network_params['n_hidden'], batch_size, lrate, _gamma, network_params['dgd_window'])

    for exp in range(1,num_exps+1):
        print('starting exp {}'.format(exp))
        st = time.time()
        
        losses_train, losses_test, grads, proc_grads = run_training((inputs_train, inputs_test), (outputs_train, outputs_test), n_epochs=epochs, learning_rate = lrate, _gamma = _gamma, tqdm_on=tqdm_on, batch_size=batch_size, control=control, network_nodes=network_nodes)
        
        elapsed = int(time.time()-st)
        times.append(elapsed)
        
        losses = np.concatenate([np.array(losses_train).reshape(-1,1), np.array(losses_test).reshape(-1,1)],axis=1)
        df = pd.DataFrame(losses)
        
        df.columns = ['loss_train','loss_test']
        df['experiment'] = exp
        df['depth'] = network_params['n_layers']
        df['width'] = network_params['n_hidden']
        df['batch_size'] = batch_size
        df['lr'] = lrate
        df['gamma'] = _gamma
        df['name'] = name
        df['dgd_window'] = network_params['dgd_window']

        if df_test is None:
            df_test = df
        else:
            df_test = pd.concat([df_test,df],axis=0)
            

    meantime = int(np.array(times).mean())
    df_test.reset_index().rename(columns={'index':'epoch'})
    df_test.to_csv('experiment_results/{}_t{}.csv'.format(name, meantime))
            
    return df_test, grads, proc_grads





def main():
    
    # load data
    
    inputs_train, inputs_test, outputs_train, outputs_test = load_saved('dde')
    
    # build neural network
    network_params, exp_params = {}, {}
    network_params['n_layers'] = 500
    network_params['n_hidden'] = 20
    network_params['dgd_window'] = 1
    
    print(network_params)
    
    network_nodes = {}
    
    network_nodes['input_layer'], network_nodes['output_truth'], network_nodes['lr'], network_nodes['gamma'], network_nodes['loss'], network_nodes['control_opt'], network_nodes['custom_opt'],  network_nodes['output_pred'], network_nodes['grads_and_vars'], network_nodes['processed_gradients'] = build_network(n_input = 13, 
                                                          n_output = 1, 
                                                          n_hidden = network_params['n_hidden'], 
                                                          n_layers = network_params['n_layers'],
                                                          ini_mean = 0,
                                                          ini_stddev = 1/(2*100),
                                                          output_type = 'linear',

                                                          dgd_window = network_params['dgd_window']
                                                         )
    
    # run experiments

    exp_params['num_exps'] = 5
    exp_params['tqdm_on'] = False

    # directional gradient delta experiments
    for lr, epochs in zip([0.1, 0.01, 0.001, 0.0001], [40, 75, 100, 500]):
    #for lr, epochs in zip([0.001],[100]):
        for batch_size in [128]:
        
            exp_params['epochs'] = epochs
            exp_params['batch_size'] = batch_size
            exp_params['lrate'] = lr
            
            # run control experiment
            exp_params['_gamma'] = 0.0
            exp_params['control'] = True  
            print(exp_params)
            
            _, _, _ = run_exps(num_exps = exp_params['num_exps'], 
                               network_params = network_params,
                               network_nodes = network_nodes,
                               tqdm_on = exp_params['tqdm_on'], 
                               epochs = exp_params['epochs'], 
                               lrate = exp_params['lrate'], 
                               _gamma = exp_params['_gamma'], 
                               batch_size = exp_params['batch_size'], 
                               control = exp_params['control'],

                               inputs_train = inputs_train, 
                               inputs_test = inputs_test,
                               outputs_train = outputs_train, 
                               outputs_test = outputs_test,
                              )


            # run experiment with gamma

            for gamma in [0.01, 0.1, 0.5, 1.0, 3.0, 9.0]:
            #for gamma in [1.0]:

                exp_params['_gamma'] = gamma
                exp_params['control'] = False
                print(exp_params)

                _, _, _ = run_exps(num_exps = exp_params['num_exps'], 
                                   network_params = network_params,
                                   network_nodes = network_nodes,
                                   tqdm_on = exp_params['tqdm_on'], 
                                   epochs = exp_params['epochs'], 
                                   lrate = exp_params['lrate'], 
                                   _gamma = exp_params['_gamma'], 
                                   batch_size = exp_params['batch_size'], 
                                   control = exp_params['control'],

                                   inputs_train = inputs_train, 
                                   inputs_test = inputs_test,
                                   outputs_train = outputs_train, 
                                   outputs_test = outputs_test,
                                  )
                
    print('done experiments!')
    
    
                
                                   
if __name__ == '__main__':
    main()