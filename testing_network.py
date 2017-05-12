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
                    filename='~model_hyperparam_cloud_test.log',
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
    # sess = tf.Session()
    # saver = tf.train.import_meta_graph(best_model_file)
    # saver.restore(sess, best_model_file)
    #
    # #am i accessing these variables correctly
    # py = tf.get_collection(model_params['classification_py'])[0]
    # cost = tf.get_collection('cost')[0]
    # X = tf.get_collection('X')[0]
    # Y = tf.get_collection('Y')[0]
    #
    tf.reset_default_graph()
    with tf.Session() as sess:
        # Load the saved model
        new_saver = tf.train.import_meta_graph(best_model_file + '.meta')

        new_saver.restore(sess, tf.train.latest_checkpoint(dirname(best_model_file)))
        #X = tf.placeholder("float", [None, 22000, 1, 4])
        #Y = tf.placeholder("float", [None, 2])
        #X = tf.get_collection('CNN_v2__X')[0]
        #Y = tf.get_collection('CNN_v2__Y')[0]

        X = tf.get_collection("_".join([model_name, '_X']))[0]
        Y = tf.get_collection("_".join([model_name, '_Y']))[0]
        py = tf.get_collection("_".join([model_name, '_py']))[0]
        cost = tf.get_collection("_".join([model_name, '_cost']))[0]
        keep_prob = tf.placeholder(tf.float32)
        print X
        print X.shape
        print Y
        print Y.shape
        print py
        print py.shape
        print cost
        print cost.shape
        print keep_prob

        ops = [cost, py]

        print type(teX[0][0])
        print type(teY[0][0])

        feed_dict={}
        feed_dict[X] = teX
        feed_dict[Y] = teY
        feed_dict[keep_prob] = 1.0


        teX_cost, teX_prob = sess.run(ops, feed_dict=feed_dict)

    return teX_cost, teX_prob

batch_size = 10
epochs = 5
input_length = 22000
num_splits = 10
test_file = '/home/Liz/all_gis_islandviewer_iv4ad_data.csv.gz'
#test_file = testing_params['test_files']

# Load in best model file
#best_model_file = join('/home/Liz/CNN_hyperparams_gpu/CNN_hyperparams_gpu_best.ckpt')
best_model_file = '/home/Liz/CNN_hyperparams_gpu_2/CNN_hyperparams_gpu_2_best.ckpt'
model_name = 'CNN_v2'
# Load the data
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
#print full_samples
batch_generator = Test_BatchGenerator(batch_size, test_file, input_length, num_splits)
teX, teY = batch_generator.next_batch(test_file)
print(teX.shape)

# Evaluate the performance
teX_cost, teX_prob  = testing(teX, teY, best_model_file)
print('test acc', np.mean(np.argmax(teY, axis=1) == np.argmax(teX_prob, axis=1)),
    'test auROC', roc_auc_score(teY, teX_prob))

logging.info('test acc', np.mean(np.argmax(teY, axis=1) == np.argmax(teX_prob, axis=1)),
    'test auROC', roc_auc_score(teY, teX_prob))

# Evaluating the performance based on majority
true_label = full_samples[0][2]
test_slice_probs = teX_prob.eval()
print len(test_slice_probs)
if (sum(i > 0.5 for i in test_slice_probs) > 5):
    test_label = 1
else:
    test_label = 0

print true_label == test_label

logging.info(true_label == test_label)
logging.info('Finished')



