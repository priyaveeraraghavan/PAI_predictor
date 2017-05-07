from batchhelper import BatchGenerator
from architecture import basic_CNN_model
import os
import tensorflow as tf, sys, numpy as np
from os.path import join,dirname,basename,exists,realpath
from os import makedirs
from sklearn.metrics import roc_auc_score

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
    with tf.Session() as sess:
        # Load the saved model
        new_graph = tf.Graph()
        new_saver = tf.train.import_meta_graph(best_model_file + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(dirname(best_model_file)))
        X = tf.get_collection('X')[0]
        Y = tf.get_collection('Y')[0]
        py = tf.get_collection('_CNN_v4_py')[0]
        cost = tf.get_collection('cost')[0]

        teX_cost, teX_prob = sess.run([cost, py], feed_dict={X: teX, Y: teY})

    #need to actually give in data for teX and teY
    #assuming have to create dictionary for testing_params like you have for training_params ?
    ops = [cost, py]
    feed_dict ={X: teX, Y: teY}

    teX_cost, teX_prob = sess.run(ops, feed_dict)

    return teX_cost, teX_prob

batch_size = 1
epochs = 10
input_length = 1000
num_splits = 10
test_file = 'first_ten_out.csv'

# Load in best model file
best_model_file = join('CNN_v4_best.ckpt')

# Load the data
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
test_batch_generator = BatchGenerator(batch_size, test_file, input_length, num_splits)
teX, teY = test_batch_generator.next_batch(test_file)
print(teX.shape)

# Evaluate the performance
teX_cost, teX_prob  = testing(teX, teY, best_model_file)
print('test acc', np.mean(np.argmax(teY, axis=1) == np.argmax(teX_prob, axis=1)),
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



