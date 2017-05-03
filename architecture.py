## Priya Veeraraghavan, Liz Martin, Sharon Wu 2017
## Defines helper functions for PAI predictor architecture construction
import tensorflow as tf
import numpy as np

class RNN:
    """A RNN model.
        
    Attributes:
        name: string name no spaces of this component of the model
        X : the tf placeholder for the input data
        Y : the tf placeholder for the labels 
        unroll_length : the number of characters on which to backprop each time
        batch_size : the size of a batch to process
        cell : GRU cell with size memory_dimension
        initial_state : tf placeholder same size as the GRU cell
        zero_state : zeros the size of the GRU cell, given the batch size. MUST be evaluated in a given session.
        final_states : concatenated output states from each of the windows read by the RNN
        logits : state probabilities following the final states determined by a fully connected output prediction layer.
        fc_layers : a list of tensors [h_fc1, h_fc2 ... logits] 
    """    
    
    def __init__(self, X, Y, size_params, **kwargs):
        self.X = tf.squeeze(X, [2])
        print "X", self.X
        self.Y = Y
        self.unroll_length = size_params['unroll_length']
        self.batch_size = size_params['batch_size']
        self.cell = tf.contrib.rnn.GRUCell(size_params['memory_dim'])
        self.initial_state = tf.placeholder(tf.float32,
                                           [self.batch_size, size_params['memory_dim']],
                                           'initial_state')
        self.zero_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.final_state = self.make_recurrent_graph()
        self.logits = self.make_output_prediction(size_params['hidden_fc'], size_params['out'])
        self.keep_prob = tf.placeholder(tf.float32)
        
        tf.add_to_collection(name+'_final_states', self.final_state)
        
    def make_recurrent_graph(self):
        x_seq_windows = int(self.X.shape[1])/self.unroll_length
        X_inp_list = [tf.split(axis=1, num_or_size_splits=self.unroll_length, value=lst) 
                      for lst in tf.split(axis=1, num_or_size_splits=x_seq_windows, value=self.X)]
        X_inp_list_corrected = [[tf.squeeze(inp_, [1]) for inp_ in x] for x in X_inp_list]
        
        ## simpler version
        #X_inp_list_corrected = [tf.squeeze(inp_, [1]) for inp_ in tf.split(axis=1, num_or_size_splits=int(self.X.shape[1]), value=self.X)]

        print "input list length", len(X_inp_list_corrected)
        print "input list first elt", X_inp_list_corrected[0]
        all_outputs = []
        all_final_states = []
        with tf.variable_scope("rnn") as scope:

            for window in X_inp_list_corrected:
                
                outputs, final_state = tf.contrib.rnn.static_rnn(self.cell, window,
                                   initial_state=self.initial_state)
                all_outputs.append(outputs)
                all_final_states.append(final_state)
                scope.reuse_variables()
        #outputs, final_state = tf.contrib.rnn.static_rnn(self.cell, X_inp_list_corrected,
        #                                                initial_state=self.initial_state)
        self.final_state = final_state
        
        return tf.concat(axis=1, values=all_final_states)
        #return final_state
    
            
    def make_output_prediction(self, hidden_fc, out):
        print self.final_state.shape
        batch, h = self.final_state.shape
        rnn_out_size = int(h)
        
        flattened_out_states = tf.reshape(self.final_states, [-1, rnn_out_size])
        W1 = tf.Variable(tf.truncated_normal([rnn_out_size, hidden_fc]), name="W1", trainable=True)
        B1 = tf.Variable(tf.zeros([1, hidden_fc]), name="B1", trainable=True)
        hidden1 = tf.nn.tanh(tf.add(tf.matmul(flattened_out_states, W1), B1))
        
        W2 = tf.Variable(tf.truncated_normal([hidden_fc, out]), name="W2", trainable=True)
        B2 = tf.Variable(tf.zeros([out]), name="B2", trainable=True)
        
        out_layer = tf.add(tf.matmul(hidden1, W2), B2)
        self.fc_layers = [hidden1, out_layer]
        self.classification_py = tf.nn.softmax(out_layer, name="classification_py")
        tf.add_to_collection(name+'_hidden_outlayer', out_layer)
        tf.add_to_collection(name+'_classification_py', self.classification_py)
        return out_layer
    
    def l2_loss(self):
        return sum([tf.nn.l2_loss(x) for x in self.fc_layers])
       

    
class CNN:

    def __init__(self, X, Y, size_params, **kwargs):
        self.X = X
        self.Y = Y
        self.keep_prob = tf.placeholder(tf.float32)
        self.model_name = size_params['model_name']
        print "made keep prob"
        self.conv_layers, self.pooled_layers, prev_layer= self.make_conv_layers(size_params['convolutional_size_params'])
        print "made conv layers"
        self.fc_layers = self.make_fc_layers(size_params['fc_size_params'], prev_layer)
        print "made fc layers", self.fc_layers
        self.classification_py = tf.nn.softmax(self.fc_layers[-1], name="classification_py")
        tf.add_to_collection("_".join([self.model_name, '_py']), self.classification_py)
        print "made classification layer"

    def make_conv_layers(self, convolutional_size_params):
        """ Takes size parameters for an arbitrary number of convolutional layers and returns properly connected conv layers.
    
        Arguments:
            convolutional_size_params: List of the following structure [[layer_name, [filter_height, filter_width, num_filters]]
        
        Returns:
            list of convolutional layers defined with truncated normal weight variables and bias variables initialized to 0.1???
           TODO DECIDE IF constant bias init is correct
        """
        conv_layers = []
        pooled_layers = []
        prev_layer = self.X
        for name, filter_params, pool_or_conv in convolutional_size_params:
            if pool_or_conv == "conv":
                [filter_height, filter_width, num_filters] = filter_params
                W_conv = tf.Variable(tf.truncated_normal([filter_height, filter_width, int(prev_layer.shape[3]), num_filters], stddev=0.1), name="W" + name)
                b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b" + name)
                h_conv = tf.nn.relu(tf.nn.conv2d(prev_layer, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv, name=name)
                conv_layers.append(h_conv)
                prev_layer = h_conv
                tf.add_to_collection("_".join([self.model_name, name]), h_conv)
            else:
                [filter_size, filter_size, stride, stride] = filter_params
                h_pool = tf.nn.max_pool(h_conv, ksize=[1, filter_size, filter_size, 1],
                        strides=[1, stride, stride, 1], padding='SAME')
                pooled_layers.append(h_pool)
                prev_layer = h_pool
                tf.add_to_collection("_".join([self.model_name, name]), h_pool)
            
        return conv_layers, pooled_layers, prev_layer
    
    
    def make_fc_layers(self, fc_size_params, prev_layer):
        fc_layers = []
        last_pooled_layer = prev_layer
        _, h, w, d = last_pooled_layer.shape
        prev_out = int(h)*int(w)*int(d)
        prev_layer = tf.reshape(last_pooled_layer, [-1, prev_out])

        for name, out_channel in fc_size_params:           
            W_fc = tf.Variable(tf.truncated_normal([prev_out, out_channel], stddev=0.1), name="W" + name)
            b_fc = tf.Variable(tf.constant(0.1, shape=[out_channel]), name="b" + name)
            h_fc = tf.nn.relu(tf.matmul(prev_layer, W_fc) + b_fc) 
            h_dropout = tf.nn.dropout(h_fc, self.keep_prob) 
            fc_layers.append(h_dropout)
            prev_layer = h_dropout
            prev_out = out_channel
            tf.add_to_collection("_".join([self.model_name, name]), h_fc)
            if name == 'out':
                W_out = tf.Variable(tf.truncated_normal([prev_out, out_channel], stddev=0.1), name="W_out")
                b_out = tf.Variable(tf.constant(0.1, shape=[out_channel]), name="b_out")
                h_out = tf.matmul(prev_layer, W_out) + b_out
                fc_layers.append(h_out)
                self.logits = h_out
                tf.add_to_collection("_".join([self.model_name, "out"]), h_out)
                break
        
        return fc_layers
    
    def l2_loss(self):
        return sum([tf.nn.l2_loss(x) for x in self.fc_layers] + 
                           [tf.nn.l2_loss(y) for y in self.conv_layers])
           
        

        
## ========== FUNCTIONS ==========================================================      
##
## -----------MODEL DEFINITIONS------------
# These definitions are imported to training.py. All other classes in this file are hidden.
def basic_CNN_model(X, Y, model_params):
    cnn = CNN(X, Y, model_params)
    l2_loss = model_params['l2']*cnn.l2_loss()
    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn.logits, labels=Y)) + l2_loss
    
    return cnn, total_loss

def basic_RNN_model(X, Y, model_params):
    rnn = RNN(X, Y, model_params)
    l2_loss = model_params['l2']*rnn.l2_loss()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn.fc_layers[-1], labels=Y))
    return rnn, loss
  
    

