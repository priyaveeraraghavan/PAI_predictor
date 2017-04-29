import csv
import numpy as np

data_file = "datafile.csv"
mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
worddim = len(mapper['A'])

acc = [] #list of accession numbers of all sequences
seqs = [] #list of 22kb sequences (both positve and negative
labels = [] # 1 for PAI, 0 otherwise

"""
Reading in accenssion numbers, sequences, and labels for positive and negative data sets
"""

#grab sequence ids, starts, and ends of pais from the file
with open(data_file, 'rb') as file:
  reader = csv.reader(file)
  print 'Reading data file...'
  for row in reader:
    acc.append(row[0])
    seqs.append(row[1])
    labels.append(row[2])

print 'Finished reading IslandViewer file.'
#remove headers
acc.pop(0)
seqs.pop(0)
labels.pop(0)

"""
Turn sequences into one-hot encodings
"""