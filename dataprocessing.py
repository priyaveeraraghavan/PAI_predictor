import csv
import numpy as np
import pandas as pd

data_file = '/Users/Liz/Downloads/all_gis_islandviewer_iv4aa_data.csv'
#data_file = "C:/Users/Sharon/Downloads/all_gis_islandviewer_iv4aa_data.csv"
mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],
          'N':[0,0,0,0],'R':[0,1,0,1],'Y':[1,0,1,0],'M':[1,1,0,0],
          'K':[0,0,1,1],'S':[0,1,1,0],'W':[1,0,0,1],'B':[0,1,1,1],
          'V':[1,1,1,0],'H':[1,1,0,1],'D':[1,0,1,1]}

worddim = len(mapper['A'])

"""Try this code"""
# Run this line first
for key, val in mapper.items():
          mapper[key] = np.array(val)
# Now run these 
inp = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=str)
getseq = lambda seq: np.expand_dims(np.concatenate([np.expand_dims(mapper[i], axis=0) for i in seq], axis=0), axis=0)
all_seqs = np.expand_dims(np.concatenate([getseq(x) for x in inp[:,1]], axis=0), axis=2)

"""
Reading in sequences for positive and negative data sets
"""
with open(data_file) as fi1e:
    seqdata = np.array([list(x.strip().split(',')[1]) for x in fi1e])
seqdata = np.delete(seqdata, (0), axis=0)
#print 'Finished reading data file.'


data = []
for l in seqdata:
    bases = []
    for char in l:
        n = mapper[char]
        bases.append(n)
    data.append(bases)

print len(data[0])
data = np.asarray(data)
print data.shape
"""
#print len(seqdata[1])
#print len(seqdata)
#print seqdata.shape


#Reading in labels for positive and negative data sets

with open(data_file) as fi1e:
    labels = np.array([list(x.strip().split(',')[2]) for x in fi1e])
#print 'Finished reading label file.'
labels = np.delete(labels, (0), axis=0)

"""
seqdata = []
labels = []
with open(data_file, 'rb') as file:
  reader = csv.reader(file)
  for row in reader:
    seqdata.append(row[1])
    labels.append(row[2])

seqdata.pop(0)
labels.pop(0)

for i in seqdata:
    new_seq = ','.join(map(str, i))
    seqdata.remove(i)
    #print new_seq
    seqdata.append(new_seq)
    #for char in i:

#print seqdata[0]
print len(seqdata)
seqdata = np.asarray(seqdata)
seqdata.shape = (4064, 1)
labels = np.asarray(labels)
print seqdata.shape

"""

Turn sequences into one-hot encodings
"""
# Function to embed sequences [batch size, seq length, 1, one hot]
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
    def embed(seq,mapper,worddim):
        return np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in seq])


    new_array = np.asarray([   [embed(seq,mapper,worddim)] for seq in data])
    new_array.shape = (len(data), len(data[1]), 1, worddim)
    print new_array.shape
    return new_array


"""
  seqdata_transformed = map(lambda i: bp_mappings(i, mapper), data)

  print seqdata_transformed[1]
  print len(data)
  print len(data[1])
  print worddim
  new_array = np.ndarray.reshape(seqdata_transformed, (len(data))) #, len(data[1]), 1, worddim))
  print new_array.shape
  return np.asarray(new_array)
"""


def bp_mappings(sample, mapper):
  return map(lambda x: mapper[x], sample)

# Embed the sequences
seqdata_transformed = seq2feature(seqdata,mapper,worddim)
print seqdata_transformed.shape
