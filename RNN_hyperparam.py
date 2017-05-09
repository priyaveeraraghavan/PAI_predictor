from training import training
import pandas as pd
import numpy as np
import tensorflow as tf
print tf.__version__
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/afs/csail.mit.edu/u/p/priyav/PAI_data/PAI_predictor/RNN_hyperparam.log',
                    filemode='w')
logging.info('Started module.')

model_type = "basic_RNN"
model_name = "RNN_hyperparam"
logging.info(model_type)
logging.info(model_name)
model_params = { 'input_shape' : [ 22000, 1, 4],                                                                                         
                 'unroll_length': 220,                                                                                                  
                 'batch_size': 50,                                                                                                     
                 'valid_size': 100,                                                                                                     
                 'memory_dim': 15,                                                                                                    
                 'hidden_fc': 10,                                                                                                      
                 'out' : 2,
                 'splits': 1,
                 'model_name': "RNN_hyperparam",
                 'l2': 0.01}  

training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_examples' : 10000,
                      'epochs' : 5,
                      'lr' : 0.01, 
                      'train_files' : ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4aa_data.csv.gz'],
                      'valid_files' : ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ag_data.csv.gz']}

perform=[]

for _dropout_keep in [1e-3,0.2,0.5,0.8]:
    training_params['dropout_keep_prob'] = _dropout_keep
    for _l2_coef in [1e-06,0.0]:
        training_params['l2'] = _l2_coef
        model_params['l2'] = _l2_coef
        for _lr in [1e-1,1e-3,1e-5]:
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
p.to_csv('/afs/csail.mit.edu/u/p/priyav/PAI_data/PAI_predictor/RNN_hyperparam.csv')
logging.info('Finished')
