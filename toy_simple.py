from training import training
import numpy as np
import tensorflow as tf
print tf.__version__
model_type = "basic_CNN"
model_name = "CNN_v2"
model_params = { 'model_name' : 'CNN_v2',
                 'input_shape' : [ 8, 1, 4],
                 'splits': 4,
                 'convolutional_size_params' : [["conv1", [1, 4, 32], 'conv'],
                                                ["pool1", [2, 2, 2, 2], 'pool']],
                 'fc_size_params' : [['h_fc1', 128],
                                     ['out', 2]],
                 'l2' : 0.001,
                 'batch_size' : 2,
                 'valid_size' : 2
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                      'epochs' : 10,
                      'lr' : 0.01, 
                      'train_files' : ['random_simple.csv'],
                      'valid_files' : ['random_simple.csv']}

training(model_name, model_type, model_params, training_params)
