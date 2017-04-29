import csv
import numpy as np

data_file = "C:/Users/Sharon/Downloads/all_gis_islandviewer_iv4aa_data.csv"
mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],
          'N':[0,0,0,0],'R':[0,1,0,1],'Y':[1,0,1,0],'M':[1,1,0,0],
          'K':[0,0,1,1],'S':[0,1,1,0],'W':[1,0,0,1],'B':[0,1,1,1],
          'V':[1,1,1,0],'H':[1,1,0,1],'D':[1,0,1,1]}
worddim = len(mapper['A'])

"""
Reading in sequences for positive and negative data sets
"""
with open(data_file) as fi1e:
    seqdata = np.asarray([list(x.strip().split(',')[1]) for x in fi1e])
print 'Finished reading data file.'
seqdata = np.delete(seqdata, (0), axis=0)
print seqdata.shape

"""
Reading in labels for positive and negative data sets
"""
with open(data_file) as fi1e:
    labels = np.asarray([list(x.strip().split(',')[2]) for x in fi1e])
print 'Finished reading label file.'
labels = np.delete(labels, (0), axis=0)
print labels.shape
"""
Turn sequences into one-hot encodings
"""
# Function to embed sequences [batch size, seq length, one hot, ???]
def seq2feature(data, mapper, worddim):
  ################################################################################
  # a function that transforms DNA sequences to matrices of shape
  # (samplesize,1,seqlength,worddim). Note that your implementation should work
  # with any sequence length.
  # - data: a 2D character array of DNA sequences of shape (samplesize,seqlength)
  # - mapper: a dictionary that encodes nucleotide character one-hot vector of (1,worddim)
  # - wordim: the size of the vector each nucleotide character is embedded to
  # - to return: a embedded array of shape (samplesize,1,seqlength,worddim)
  ################################################################################
  seqdata_transformed = np.asarray(map(lambda i: bp_mappings(i, mapper), data))
  print 'shape:' + str(seqdata_transformed.shape)
  print seqdata_transformed[1]
  return np.asarray(np.reshape(seqdata_transformed, (len(data), len(data[1]), worddim, 1)))


def bp_mappings(sample, mapper):
  return map(lambda x: mapper[x], sample)

# Embed the sequences
seqdata_transformed = seq2feature(seqdata,mapper,worddim)
print seqdata_transformed.shape
