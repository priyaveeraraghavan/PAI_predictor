# imports
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from os.path import join,dirname
import time
import tensorflow as tf
from test_batchhelper import Test_BatchGenerator

#this is pretty much ripped straight from pset 1. Need to formulate it for our architecture
%matplotlib inline
def plot_filter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(30,20))
    n_columns = 3
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(np.transpose(units[0,:,:,i]), interpolation="nearest", cmap="afmhot")


# Loading in the best model to visualize filters
# h_conv1, h_conv2 are the outputs of your convolutional layers
# ex. h_conv1 = tf.nn.relu(conv2d(x, W) + b)
tf.reset_default_graph()

best_model_file = 'CNN_hyperparams_gpu_2_best.ckpt'
model_name = 'CNN_v2'
sess = tf.Session()
new_saver = tf.train.import_meta_graph(best_model_file + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
conv1 = tf.get_collection('_'.join([model_name, 'conv1']))
conv2 = tf.get_collection('_'.join([model_name, 'conv2']))

# We will get the convolutional layer activations for a specific sample
# Load in data
test_file = 'all_gis_islandviewer_iv4aa_data_half.gz'
batch_size = 10
input_length = 22000
num_splits = 10
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
full_samples_neg = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[1999:2009]
test_batch_generator = Test_BatchGenerator(batch_size, test_file, input_length, num_splits)
teX, teY = test_batch_generator.next_batch(test_file)

# Pick a particular sample to look at filters for
samp = 0
teX_samp = teX[samp]
teX_samp = np.expand_dims(teX_samp, axis=0)
teY_samp = teY[samp]
teY_samp = np.expand_dims(teY_samp, axis=0)

X = tf.get_collection("_".join([model_name, '_X']))[0]
Y = tf.get_collection("_".join([model_name, '_Y']))[0]
keep_prob = tf.placeholder(tf.float32)
feed_dict={}
feed_dict[X] = teX_samp
feed_dict[Y] = teY_samp
feed_dict[keep_prob] = 1.0
img_filters1, img_filters2 = sess.run([conv1, conv2], feed_dict=feed_dict)

# Show the activations of the first convolutional filters for the first test sample
plot_filter(img_filters1)

# Show the activations of the second convolutional filters for the first test sample
plot_filter(img_filters2)

#Visualization Filter experiment
# def getActivations(layer, samp_seq, samp_label):
#   feed_dict={}
#   feed_dict[model.X] = samp_seq
#   feed_dict[model.Y] = samp_label
#   feed_dict[model.keep_prob] = 1.0
#   units = sess.run(layer,feed_dict=feed_dict)
#   plot_filter(units)
# getActivations(img_filters1, teX_samp, teY_samp)
#getActivations(img_filters2, teX_samp, teY_samp)
#plt.save_fig("orig_img1.png", bbox_inches='tight')
