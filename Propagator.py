import numpy as np
import tensorflow as tf

from TFEnvironment import TFEnvironment
from SimpleRegressor import SimpleRegressor

class Propagator(TFEnvironment):

    def __init__(self, name):
        super(Propagator, self).__init__()
        self.name = name

    def generate_propagator(self, from_morphism, to_morphism, analysis_profile, est, num_batches = 40000, batchsize = 300, debug = False):
        """
        Builds the propagator, i.e. trains the neural network to compute the reweighting factors.
        """

        # build the TF graph
        self.build()
        
        # first, extract the training data from the analysis profile: only need to take the output of the 'from_morphism' as well as the output of the 'to_morphism'
        from_morphism_data = np.array(analysis_profile.profile[from_morphism])
        to_morphism_data = np.array(analysis_profile.profile[to_morphism])
        densrat_source = est.evaluate(np.array(from_morphism_data))

        assert(np.shape(to_morphism_data) == np.shape(densrat_source))

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        
        # now can start the training
        for batch in range(num_batches):
            inds = np.random.choice(len(densrat_source), batchsize)

            densrat_source_step = densrat_source[inds]
            to_morphism_data_step = to_morphism_data[inds]

            if debug:
                print("target_pos = {}".format(to_morphism_data_step))
                print("DR_source = {}".format(densrat_source_step))
                print("DR_target = {}".format(self.predict(to_morphism_data_step)))
            
            self.train_step(to_morphism_data_step, densrat_source_step)

            if not batch % 1000:
                print("regression loss = {}".format(self.evaluate_regression_loss(to_morphism_data_step, densrat_source_step)))
        
    def build(self):

        self.reg = SimpleRegressor(self.name + "_reg", hyperpars = {"num_layers": 2, "num_units": 30})
        
        with self.graph.as_default():
            # only need the final variables, as well as the initial likelihood ratio (evaluated at the corresponding inputs) as inputs
            self.to_morphism_data_in = tf.placeholder(tf.float32, [None, 1], name = 'to_morphism_data_in')
            self.densrat_source_in = tf.placeholder(tf.float32, [None, 1], name = 'densrat_source_in')

            self.densrat_target, self.reg_vars = self.reg.build_model(self.to_morphism_data_in)
            
            # assemble the regression loss
            self.regression_loss = tf.reduce_mean(tf.math.square(self.densrat_source_in - self.densrat_target), axis = 0)

            # set up the optimizer
            self.opt = tf.train.AdamOptimizer(learning_rate = 1e-4,
                                              beta1 = 0.5,
                                              beta2 = 0.3,
                                              epsilon = 1e-6)
            
            self.train_regressor = self.opt.minimize(self.regression_loss, var_list = self.reg_vars)

    def train_step(self, to_morphism_data_step, densrat_source_step):
        with self.graph.as_default():
            self.sess.run([self.train_regressor], feed_dict = {self.to_morphism_data_in: to_morphism_data_step, self.densrat_source_in: densrat_source_step})

    def evaluate_regression_loss(self, to_morphism_data, densrat_source):
        with self.graph.as_default():
            return self.sess.run(self.regression_loss, feed_dict = {self.to_morphism_data_in: to_morphism_data, self.densrat_source_in: densrat_source})
            
    def predict(self, val):      
        with self.graph.as_default():
            retval = self.sess.run(self.densrat_target, feed_dict = {self.to_morphism_data_in: val}).flatten()

        return retval
