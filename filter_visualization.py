# imports
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from os.path import join,dirname
import time
import tensorflow as tf
from batchhelper import BatchGenerator

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
best_model_file = 'CNN_v4_best.ckpt.'
model_name = 'CNN_v4'
sess = tf.Session()
new_saver = tf.train.import_meta_graph(best_model_file + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(dirname(best_model_file)))
conv1 = tf.get_collection('_'.join(model_name, 'conv1'))
conv2 = tf.get_collection('_'.join(model_name, 'conv2'))

# We will get the convolutional layer activations for a specific sample
# Load in data
test_file = 'first_ten_out.csv'
batch_size = 10
input_length = 22000
num_splits = 10
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
test_batch_generator = BatchGenerator(batch_size, test_file, input_length, num_splits)
teX, teY = test_batch_generator.next_batch(test_file)
print(teX.shape)

# Pick a particular sample to look at filters for
samp = 0
teX_samp = teX[samp,1]
teY_samp = teY[samp,2]

img_filters1, img_filters2 = sess.run([conv1, conv2], feed_dict={X: teX_samp, Y: teY_samp})

# Show the activations of the first convolutional filters for the first test sample
plot_filter(img_filters1)

# Show the activations of the second convolutional filters for the first test sample
plot_filter(img_filters2)

#Visualization Filter experiment
def getActivations(layer, samp_seq, samp_label):
     units = sess.run(layer,feed_dict={X: samp_seq, Y: samp_label})
     plot_filter(units)
getActivations(conv1, teX_samp, teY_samp)
getActivations(conv2, teX_samp, teY_samp)
#plt.save_fig("orig_img1.png", bbox_inches='tight')
