from training import training
import numpy as np
import tensorflow as tf
print tf.__version__
model_type = "basic_CNN"
model_name = "CNN_v4"
model_params = { 'model_name' : 'CNN_v2',
                 'input_shape' : [ 22000, 1, 4],
                 'splits' : 10,
                 'convolutional_size_params' : [["conv1", [1, 10, 32], 'conv'],
                                                ["pool1", [2, 2, 2, 2], 'pool'],
                                                ["conv2", [1, 10, 64], 'conv'],
                                                ["pool2", [2, 2, 2, 2], 'pool']],
                 'fc_size_params' : [['h_fc1', 10],
                                     ['out', 2]],
                 'l2' : 0.00001,
                 'batch_size' : 20,
                 'valid_size' : 15
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                      'epochs' : 500,
                      'lr' : 0.01, 
                      'train_files' : ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4aa_data.csv.gz',
                                       '/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ad_data.csv.gz',
                                       '/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ae_data.csv.gz'],
                      'valid_files' : ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ag_data.csv.gz']}

training(model_name, model_type, model_params, training_params)
