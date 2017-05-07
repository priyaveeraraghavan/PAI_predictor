from batchhelper import BatchGenerator
from architecture import basic_CNN_model
import os
import tensorflow as tf
import numpy as np

def testing(model_params, best_model_file):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(best_model_file)
    saver.restore(sess, best_model_file)

    #am i accessing these variables correctly
    py = tf.get_collection(model_params['classification_py'])[0]
    cost = tf.get_collection('cost')[0]
    X = tf.get_collection('X')[0]
    Y = tf.get_collection('Y')[0]


    #need to actually give in data for teX and teY
    #assuming have to create dictionary for testing_params like you have for training_params ?
    ops = [cost, py]
    feed_dict ={X: teX, Y: teY}

    teX_cost, teX_prob = sess.run(ops, feed_dict)

    return teX_cost, teX_prob

teX_cost, teX_prob  = testing(_teX, _teY, _best_model_file)
print('test acc', np.mean(np.argmax(_teY, axis=1) == np.argmax(teX_prob, axis=1)),
    'test auROC', roc_auc_score(_teY, teX_prob))

