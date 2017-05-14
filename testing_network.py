from test_batchhelper import Test_BatchGenerator
from architecture import basic_CNN_model
import os
import tensorflow as tf, sys, numpy as np
from os.path import join,dirname,basename,exists,realpath
from os import makedirs
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='~testing_CNN_hyperparams_final.log',
                    filemode='w')
logging.info('Started module.')

testing_params =  { 'dropout_keep_prob' : 1.0,
                     'max_examples' : 10000,
                      'epochs' : 5,
                      'lr' : 0.01,
                      'test_files' : ['/home/Liz/all_gis_islandviewer_iv4ad_data.csv.gz']}

def testing(teX, teY, best_model_file):
    ################################################################################
    # Arguments:
    # - teX: embedded sequences of test set
    # - teY: label of test set
    # - best_model_file: the path of the best trained model
    # Values:
    # - teX_cost: the total loss on test set
    # - teX_prob: the probability of each label for each sample
    ################################################################################

    # Load the saved model
    sess= tf.Session()
    new_saver = tf.train.import_meta_graph(best_model_file + '.meta')

    new_saver.restore(sess, tf.train.latest_checkpoint(dirname(best_model_file)))

    X = tf.get_collection("_".join([model_name, '_X']))[0]
    Y = tf.get_collection("_".join([model_name, '_Y']))[0]
    py = tf.get_collection("_".join([model_name, '_py']))[0]
    cost = tf.get_collection("_".join([model_name, '_cost']))[0]
    conv1 = tf.get_collection('_'.join([model_name, 'conv1']))[0]
    conv2 = tf.get_collection('_'.join([model_name, 'conv2']))[0]
    keep_prob = tf.get_collection("_".join([model_name, '_keep_prob']))[0]

    ops = [cost, py]


    feed_dict={}
    feed_dict[X] = teX
    feed_dict[Y] = teY
    feed_dict[keep_prob] = 1.0

    teX_cost, teX_prob = sess.run(ops, feed_dict=feed_dict)

    return teX_cost, teX_prob


tf.reset_default_graph()
batch_size = 10
epochs = 5
input_length = 22000
num_splits = 10
test_file = '/home/Liz/all_gis_islandviewer_iv4ad_data.csv.gz'
#test_file = testing_params['test_files']


# Load in best model file
#best_model_file = join('/home/Liz/CNN_hyperparams_gpu/CNN_hyperparams_gpu_best.ckpt')
best_model_file = '/home/Liz/CNN_hyperparams_final/CNN_hyperparams_final_best.ckpt'
model_name = 'CNN_hyperparams_final'
# Load the data
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
#print full_samples
test_batch_generator = Test_BatchGenerator(batch_size, test_file, input_length, num_splits)
teX, teY = test_batch_generator.next_batch(test_file)


teX_cost, teX_prob  = testing(teX, teY, best_model_file)

teY_binary = np.round(np.expm1(teY))


# Evaluating the performance based on majority
true_label = full_samples[0][2]
test_slice_probs = teX_prob
if (sum(i > 0.5 for i in test_slice_probs) > 5).all():
    test_label = 1.0
else:
    test_label = 0.0
print true_label
print true_label == test_label

# Evaluate the performance

test_acc = np.mean(np.argmax(teY, axis=1) == np.argmax(teX_prob, axis=1))
print test_acc
auROC = roc_auc_score(teY_binary, teX_prob)
print auROC

print('test acc', test_acc), ('test auROC',auROC)

logging.info('test acc %s' % test_acc)

logging.info('test auROC %s' % auROC)


logging.info(true_label == test_label)
logging.info('Finished')



