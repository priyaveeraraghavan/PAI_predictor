from test_batchhelper import Test_BatchGenerator
from architecture import basic_CNN_model
import os
import tensorflow as tf, sys, numpy as np
from os.path import join,dirname,basename,exists,realpath
from os import makedirs
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='~testing_CNN_new.log',
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
    #conv2 = tf.get_collection('_'.join([model_name, 'conv2']))[0]
    keep_prob = tf.get_collection("_".join([model_name, '_keep_prob']))[0]

    ops = [cost, py]


    feed_dict={}
    feed_dict[X] = teX
    feed_dict[Y] = teY
    feed_dict[keep_prob] = 1.0

    teX_cost, teX_prob = sess.run(ops, feed_dict=feed_dict)

    return teX_cost, teX_prob


tf.reset_default_graph()
batch_size = 100
epochs = 5
input_length = 22000
num_splits = 10
test_file = '/home/Liz/all_gis_islandviewer_iv4ad_data.csv.gz'
#test_file = testing_params['test_files']


# Load in best model file
#best_model_file = join('/home/Liz/CNN_hyperparams_gpu/CNN_hyperparams_gpu_best.ckpt')
best_model_file = '/home/Liz/CNN_new_arch/CNN_new_arch_best.ckpt'
model_name = 'CNN_new_arch'
# Load the data
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
#print full_samples
test_batch_generator = Test_BatchGenerator(batch_size, test_file, input_length, num_splits)
teX, teY = test_batch_generator.next_batch(test_file)
print teY[:,0]
teY_flat = [-1 if x==0.1 else 1 for x in teY[:,0]]
teX_cost, teX_prob  = testing(teX, teY, best_model_file)

#print teY

teY_binary = np.round((teY))
#print teY_binary

teX_prob_binary = np.round((teX_prob))
#print teX_prob_binary
# Evaluating the performance based on majority
true_labels = full_samples[:,2]
test_slice_probs = teX_prob
test_labels = []
for samp in range(len(true_labels)):
    start = samp * num_splits
    end = (samp+1) * num_splits
    count = 0
    for split in test_slice_probs[start:end]:
        if split.all() >= 0.5:
            count += 1
    if count >= 5:
        test_labels.append('1')
    else:
        test_labels.append('0')
#print len(true_labels)
#print len(test_labels)


# Evaluate the performance

test_acc = np.mean(np.argmax(teY_binary, axis=1) == np.argmax(teX_prob, axis=1))
print test_acc
auROC = roc_auc_score(teY_binary, teX_prob)
print auROC

print('test acc', test_acc), ('test auROC',auROC)

logging.info('test acc %s' % test_acc)

logging.info('test auROC %s' % auROC)

prob_PAI = teX_prob[:, 0]
prob_notPAI = teX_prob[:, 1]

#teX_prob flipped: [prob not a PAI, prob a PAI]
output = np.concatenate([np.expand_dims(teY_flat, axis=1), np.expand_dims(prob_notPAI,axis=1), np.expand_dims(prob_PAI,axis=1)], axis=1)
print output
logging.info('Finished')
np.savetxt('~prob_predictions_new.txt', output, header='True Label,-1,1', fmt="%.3f")



