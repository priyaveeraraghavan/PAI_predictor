import tensorflow as tf
from training import training
import pandas as pd
import numpy as np
print tf.__version__
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='~model_hyperparam_cloud3.log',
                    filemode='w')
logging.info('Started module.')

model_type = "basic_CNN"
model_name = "CNN_hyperparams_gpu_3"
logging.info(model_type)
logging.info(model_name)
model_params = { 'model_name' : 'CNN_hyperparams_gpu_3',
                 'input_shape' : [ 22000, 1, 4],
                 'splits': 10,
                 'convolutional_size_params' : [["conv1", [1, 10, 32], 'conv'],
                                                ["pool1", [5, 5, 5, 5], 'pool'],
                                                ["conv2", [1, 10, 32], 'conv'],
                                                ["pool2", [5, 5, 5, 5], 'pool']],
                 'fc_size_params' : [['h_fc1', 128],
                                     ['out', 2]],
                 'l2' : 0.001,
                 'batch_size' : 10,
                 'valid_size' : 10
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_examples' : 10000,
                      'epochs' : 5,
                      'lr' : 0.01, 
                      'train_files' : ['/home/Liz/all_gis_islandviewer_iv4aa_data.csv.gz'],
                      'valid_files' : ['/home/Liz/all_gis_islandviewer_iv4ag_data.csv.gz']}

perform=[]

for _dropout_keep in [0.5,0.8]:
    training_params['dropout_keep_prob'] = _dropout_keep
    for _l2_coef in [1e-03,1e-06]:
        training_params['l2'] = _l2_coef
        for _lr in [1e-1,1e-3]:
            training_params['lr'] = _lr
            _best_cost = training(model_name, model_type, model_params, training_params)
            msg = 'dropout_keep:',_dropout_keep, 'l2 coef:',_l2_coef, 'lr:', _lr, 'Best val loss',_best_cost
            print msg
            logging.info(msg)
            perform.append([_dropout_keep, _l2_coef, _lr, _best_cost])   

logging.info('Finished with iterations.')
p =  pd.DataFrame(perform, columns=['dropout_keep','l2_coeff','lr','val_loss']).pivot_table(values=['val_loss'],columns=['l2_coeff'],index=['dropout_keep','lr'])

print p
logging.info('Writing to csv')
p.to_csv('~model_hyperparams_cloud3.csv')
logging.info('Finished')
