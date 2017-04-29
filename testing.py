import csv
import numpy as np

data_file = "C:/Users/Sharon/Downloads/all_gis_islandviewer_iv4aa_data.csv"
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
inp = np.loadtxt(data_file, delimiter=',', skiprows=1, dtype=str)
print len(inp)



lens = np.expand_dims(map(lambda x: int(len(x)), inp[:200,1]), axis=1)
in2 = np.concatenate([inp[:200], lens], axis=1)
filtered = in2[in2[:200,3] == '22000']
print len(filtered)

getseq = lambda seq: np.expand_dims(np.concatenate([np.expand_dims(mapper[i], axis=0) for i in seq], axis=0), axis=0)
all_seqs = np.expand_dims(np.concatenate([getseq(x) for x in filtered[:,1]], axis=0), axis=2)

print all_seqs.shape

# """
# Reading in sequences for positive and negative data sets
# """
# seqdata = []
# with open(data_file) as file:
#     reader = csv.reader(file)
#     for row in reader:
#         seq = row[1]
#         seqlist = row[1].strip().split()
#         seqdata.append([list(bp) for bp in seqlist])
#     #seqdata = np.asarray([list(x.strip().split(',')[1]) for x in file])
#     seqdata = np.asarray(seqdata)
# print 'Finished reading data file.'
# seqdata = np.delete(seqdata, (0), axis=0)
# print seqdata.shape
# print seqdata[1:5]
#
# """
# Reading in labels for positive and negative data sets
# """
# with open(data_file) as file:
#     labels = np.asarray([list(x.strip().split(',')[2]) for x in file])
# print 'Finished reading label file.'
# labels = np.delete(labels, (0), axis=0)
# print labels.shape
# """
# Turn sequences into one-hot encodings
# """
# # Function to embed sequences [batch size, seq length, 1, one hot]
# def seq2feature(data, mapper, worddim):
#   ################################################################################
#   # a function that transforms DNA sequences to matrices of shape
#   # (samplesize,1,seqlength,worddim). Note that your implementation should work
#   # with any sequence length.
#   # - data: a 2D character array of DNA sequences of shape (samplesize,seqlength)
#   # - mapper: a dictionary that encodes nucleotide character one-hot vector of (1,worddim)
#   # - wordim: the size of the vector each nucleotide character is embedded to
#   # - to return: a embedded array of shape (samplesize,1,seqlength,worddim)
#   ################################################################################
#   seqdata_transformed = map(lambda i: bp_mappings(i, mapper), data)
#   return np.asarray(np.reshape(seqdata_transformed, (len(data), len(data[1]), 1, worddim)))
#
#
# def bp_mappings(sample, mapper):
#   return map(lambda x: mapper[x], sample)
#
# # Embed the sequences
# seqdata_transformed = seq2feature(seqdata,mapper,worddim)
# print seqdata_transformed.shape
