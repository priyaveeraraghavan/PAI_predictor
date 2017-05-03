from training import training
import numpy as np
import tensorflow as tf
print tf.__version__
model_type = "basic_CNN"
model_name = "CNN_v2"
model_params = { 'model_name' : 'CNN_v2',
                 'input_shape' : [ 2200, 1, 4],
                 'splits': 10,
                 'convolutional_size_params' : [["conv1", [1, 10, 32], 'conv'],
                                                ["pool1", [5, 5, 5, 5], 'pool'],
                                                ["conv2", [1, 10, 32], 'conv'],
                                                ["pool2", [5, 5, 5, 5], 'pool']],
                 'fc_size_params' : [
                                     ['out', 2]],
                 'l2' : 0.00001,
                 'batch_size' : 10,
                 'valid_size' : 10
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                      'epochs' : 100,
                      'lr' : 0.001, 
                      'train_files' : ['toy_large_motif_train.csv'],
                      'valid_files' : ['toy_large_motif_valid.csv']}

print 'training set', training_params['train_files']
print 'validation set', training_params['valid_files']
training(model_name, model_type, model_params, training_params)
