from training import training
import numpy as np
import tensorflow as tf
print tf.__version__
model_type = "basic_CNN"
model_name = "CNN_v2"
model_params = { 'model_name' : 'CNN_v2',
                 'input_shape' : [ 22000, 1, 4],
                 'convolutional_size_params' : [["conv1", [1, 10, 32], 'conv'],
                                                ["pool1", [5, 5, 5, 5], 'pool'],
                                                ["conv2", [1, 10, 32], 'conv'],
                                                ["pool2", [5, 5, 5, 5], 'pool'],
                                                ["conv3", [1, 10, 32], 'conv'],
                                                ["pool3", [5, 5, 5, 5], 'pool']],
                 'fc_size_params' : [['h_fc1', 32],
                                     ['h_fc2', 32],
                                     ['out', 2]],
                 'l2' : 0.01,
                 'batch_size' : 128,
                 'valid_size' : 64
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                      'epochs' : 10,
                      'lr' : 0.01, 
                      'train_files' : ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4aa_data.csv.gz',
                                       '/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ad_data.csv.gz',
                                       '/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ae_data.csv.gz'],
                      'valid_files' : ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ag_data.csv.gz']}

training(model_name, model_type, model_params, training_params)