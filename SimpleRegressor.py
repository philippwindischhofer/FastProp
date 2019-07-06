import tensorflow as tf
import tensorflow.contrib.layers as layers

class SimpleRegressor:

    def __init__(self, name, hyperpars):
        print("initializing a SimpleRegressor")
        self.name = name
        self.hyperpars = hyperpars
                                    
    def build_model(self, in_tensor):
        with tf.variable_scope(self.name):
            lay = in_tensor
            
            for layer in range(int(float(self.hyperpars["num_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])))
                
            outputs = layers.linear(lay, 1)
                
            these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
                
        return outputs, these_vars
                                                                                                                            
