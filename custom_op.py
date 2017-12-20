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


# the optimizer


class GradientProcessor(object):
    """
    
    """
    
    def __init__(self,shape, gamma, window):
        self.shape = shape
        self.epsilon = 1e-5
        self.gamma = gamma
        self.window = window
        
        self.grad_hist1 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.para_hist1 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.grad_hist2 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.para_hist2 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.grad_hist3 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.para_hist3 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.grad_hist4 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        self.para_hist4 = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        
        self.mock_zeros = rr.ResourceVariable(tf.zeros(shape=shape,dtype=tf.float32))
        
    def _enqueue(self, new_grad, new_para):
        
        old_grad_hist1 = self.grad_hist1.read_value()
        old_para_hist1 = self.para_hist1.read_value()
        old_grad_hist2 = self.grad_hist2.read_value()
        old_para_hist2 = self.para_hist2.read_value()
        old_grad_hist3 = self.grad_hist2.read_value()
        old_para_hist3 = self.para_hist2.read_value()
        
        with tf.control_dependencies([old_grad_hist1, old_grad_hist2, old_para_hist1, old_para_hist2, old_para_hist3, old_para_hist3]):
            assign_gh4 = tf.assign(self.grad_hist4, old_grad_hist3)
            assign_ph4 = tf.assign(self.para_hist4, old_para_hist3)
            assign_gh3 = tf.assign(self.grad_hist3, old_grad_hist2)
            assign_ph3 = tf.assign(self.para_hist3, old_para_hist2)
            assign_gh2 = tf.assign(self.grad_hist2, old_grad_hist1)
            assign_ph2 = tf.assign(self.para_hist2, old_para_hist1)
            assign_gh1 = tf.assign(self.grad_hist1, new_grad)
            assign_ph1 = tf.assign(self.para_hist1, new_para)
            
        return assign_gh2, assign_ph2, assign_gh1, assign_ph1
        
    def _calc_dist(self, final, initial):
        return final - initial
    
    def _calc_norm(self, vector):
        return tf.linalg.norm(vector, ord='euclidean')
        
    def calc_deltagrad(self, fin_grad, ini_grad):
        return self._calc_dist(fin_grad, ini_grad)
    
    def calc_stepsize(self, fin_para, ini_para):
        deltapara = self._calc_dist(fin_para, ini_para)
        delta_s = self._calc_norm(deltapara) + self.epsilon # must add epsilon for numerical stability
        return deltapara, delta_s
    
    def calc_dirdiv(self, deltagrad, stepsize):
        return tf.divide(deltagrad,stepsize)
        
    def calc_proj(self, cur_grad, deltapara, stepsize):
        return tf.tensordot(tf.reshape(cur_grad,[-1]), tf.reshape(deltapara,[-1]),axes=1) / stepsize
    
    def calc_x(self, proj, dirdiv):
        return proj * dirdiv
    
    def calc_gamma(self, cur_para, fin_para, ini_para):
        """
        This is designed so that both distances have to be small in order for 
        the second order effect to be significant. If the distance between final and initial
        is large then the second order approxmiation will be very bad. If the distance between
        current and initial is large than even if the curvature approxmiation at initial is accurate,
        it may not apply at the current position (i.e. 3rd and higher order effects).
        
        NOTE: This did not work due to numerical issues. The distances are so small that this
        number blows up the update.
        
        gamma = 1/|d1| * 1/|d2|
            - If cur_para is fin_para then gamma = 1/(d^2).
            - I'd ideally avoid using the Euclidean distance so that this is more scale-invariant;
            the L2 norm would be dominated by relatively small changes in large value params
            (though I'm not sure if this succeeds)
        """
        dist_eval = self._calc_dist(fin_para, ini_para)
        dist_cur = self._calc_dist(cur_para, ini_para)
        return self._calc_norm(tf.multiply((1 / (1 + tf.abs(dist_eval))),(1 / (1 + tf.abs(dist_cur))))) + self.epsilon
    
    def calc_combined_x(self, xs):
        return tf.reduce_sum(tf.stack(xs))
    
    def calc_update(self, cur_grad, x, gamma):
        return cur_grad + tf.multiply(x, gamma)
    
    def calc_dist(self, cur_para, ini_para):
        return self._calc_norm(self._calc_dist(cur_para, ini_para))
    
    def subprocess(self, cur_grad, cur_para, fin_grad, fin_para, ini_grad, ini_para):
        deltagrad = self.calc_deltagrad(fin_grad, ini_grad)
        deltapara, delta_s = self.calc_stepsize(fin_para, ini_para)
        dirdiv = self.calc_dirdiv(deltagrad, delta_s)
        proj = self.calc_proj(cur_grad, deltapara, delta_s)
        x = self.calc_x(proj, dirdiv)
        dist = self.calc_dist(cur_para, ini_para)
        return x, dist + delta_s
    
    def process(self, grad, para):
            
        queued = self._enqueue(grad, para)
        with tf.control_dependencies(queued):
            x_12, d_12 = self.subprocess(self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist2.read_value(), self.para_hist2.read_value())
            x_13, d_13 = self.subprocess(self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist3.read_value(), self.para_hist3.read_value())
            x_23, d_23 = self.subprocess(self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist2.read_value(), self.para_hist2.read_value(), self.grad_hist3.read_value(), self.para_hist3.read_value())
            x_14, d_14 = self.subprocess(self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist4.read_value(), self.para_hist4.read_value())
            x_34, d_34 = self.subprocess(self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist3.read_value(), self.para_hist3.read_value(), self.grad_hist4.read_value(), self.para_hist4.read_value())
            x_24, d_24 = self.subprocess(self.grad_hist1.read_value(), self.para_hist1.read_value(), self.grad_hist2.read_value(), self.para_hist2.read_value(), self.grad_hist4.read_value(), self.para_hist4.read_value())
      
        xs = [x_12]
        if self.window > 1:
            xs.append(x_13)
            xs.append(x_23)
        if self.window > 2:
            xs.append(x_14)
            xs.append(x_24)
            xs.append(x_34)
        with tf.control_dependencies(xs):
            combined_x = self.calc_combined_x(xs)
        
        processed_grad = self.calc_update(grad, combined_x, self.gamma)
        return processed_grad

        
def process_gradients(processors, grads_and_vars):
    return [(processor.process(gv[0],gv[1]), gv[1]) for processor, gv in zip(processors,grads_and_vars)] 

