import tensorflow as tf
from test_batchhelper import Test_BatchGenerator
from architecture import basic_CNN_model
import numpy as np, sys
from os.path import join, exists,dirname,realpath
from os import makedirs
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize


# Function mapping from one-hot encoding to nucleotide character
nuc = ['A','C','G','T']
def mapper(vec):
    r = np.where(vec==1)[0]
    return 'N' if len(r)==0 else nuc[r[0]]

def getX(x,pos,wind,padding):
    #dimensions of x should be [2200, 4]
    out = []
    print x
    for _pos in pos:
        t_o = x[max(0, _pos - wind):min(len(x),_pos+wind+1),:]
        if wind > _pos:
            t_o = np.concatenate(([padding] * (wind-_pos),t_o))
        if _pos + wind + 1 > len(x):
            t_o = np.concatenate((t_o , [padding] * ( _pos + wind + 1 - len(x))))
        out.append(''.join([mapper(item) for item in t_o]))
    return out

def getActivatingSeq(best_model_file, batch_X):
    sess = tf.Session()
    new_graph = tf.Graph()
    new_saver = tf.train.import_meta_graph(best_model_file+'.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(dirname(best_model_file)))
    X = tf.get_collection("_".join([model_name, '_X']))[0]
    conv2 = tf.get_collection('_'.join([model_name, 'conv2']))[0]

    l1_act = np.swapaxes(np.swapaxes(sess.run(conv2, \
                                    feed_dict={X: batch_X}).squeeze(),0,2),1,2)
    l1_act_max = l1_act.reshape(32,-1).max(axis=1)

    print l1_act
    print l1_act_max
    print np.nonzero(l1_act)
    return l1_act, l1_act_max

tf.reset_default_graph()
batch_size = 10
epochs = 5
input_length = 22000
num_splits = 10
test_file = '/home/Liz/all_gis_islandviewer_iv4aa_data.csv.gz'


x_padding = [0,0,0,0]

best_model_file = join('motif_disc', '/home/Liz/CNN_hyperparams_final/CNN_hyperparams_final_best.ckpt')
model_name = 'CNN_hyperparams_final'
# Load the data
full_samples = np.loadtxt(test_file, delimiter=',', skiprows=1, dtype=str)[0:10]
#print full_samples
test_batch_generator = Test_BatchGenerator(batch_size, test_file, input_length, num_splits)
batch_X, batch_Y = test_batch_generator.next_batch(test_file)


# Get the output of the convolutional layer
l1_act, l1_act_max = getActivatingSeq(best_model_file, batch_X)

# Find all the sequences that activate any filter
with open(join('/home/Liz/CNN_hyperparams_final/','model.seqs'),'w') as f:

    # Iterate through filters to find all the sequence patches that activate each filter
    for filter_idx, _l1_act_filt in enumerate(l1_act): # shape = (?, 101)
        t_filter_act_list = []

        # Iterate through training samples
        for sample_idx, _l1_act_samp in enumerate(_l1_act_filt): # shape = (101,)

            t_x = batch_X[sample_idx,:,:,:].squeeze()


            # The filter is considered "Activated" if the output is more than half of the
            # maximum activation for this filter
            t_act_pos = np.where(_l1_act_samp > l1_act_max[filter_idx] * 0.5)[0]
            if len(t_act_pos)>0:
                #i don't think 11/2 is right for our code
                t_filter_act_list += getX(t_x,t_act_pos, 5, x_padding)

        # Output all the activating sequences
        for x in t_filter_act_list:
            f.write('%s\t%d\n' % (x,filter_idx))