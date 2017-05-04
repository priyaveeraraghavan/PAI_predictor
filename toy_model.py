from training import training
import numpy as np
import tensorflow as tf
print tf.__version__
model_type = "basic_CNN"
model_name = "toy_2_motif"
model_params = { 'model_name' : 'CNN_v2',
                 'input_shape' : [ 2200, 1, 4],
                 'splits': 10,
                 'convolutional_size_params' : [["conv1", [1, 10, 32], 'conv'],
                                                ["pool1", [2, 2, 2, 2], 'pool'],
                                                ["conv2", [1, 10, 64], 'conv'],
                                                ["pool2", [2, 2, 2, 2], 'pool'],
                                                ["conv3", [1, 10, 64], 'conv'],
                                                ["pool3", [2, 2, 2, 2], 'pool']                                            ],
            
                 'fc_size_params' : [['h_fc1', 10],
                                     ['out', 2]],
                 'l2' : 0.000001,
                 'batch_size' : 100,
                 'valid_size' : 20
               }
training_params =  { 'dropout_keep_prob' : .8,
                     'max_grad' : 0.01,
                      'epochs' : 100,
                      'lr' : 0.00001, 
                      'train_files' : ['toy_2_motif_train.csv'],
                      'valid_files' : ['toy_2_motif_valid.csv']}

print 'training set', training_params['train_files']
print 'validation set', training_params['valid_files']
training(model_name, model_type, model_params, training_params)
