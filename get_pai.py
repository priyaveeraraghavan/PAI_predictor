import sys
from Bio import Entrez
from Bio import SeqIO
from random import randint
import csv
import urllib2
import numpy as np
#import sklearn.mixture
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
#import argparse

Entrez.email = 'eemartin14@gmail.com'
#islandviewer_dir = '/Users/Liz/Downloads/all_gis_islandviewer_iv4.csv'
islandviewer_dir = 'C:/Users/Sharon/Documents/MIT/Senior Year/6.802/Final Project/all_gis_islandviewer_iv4.csv'
islandviewer_dir = '/afs/csail.mit.edu/u/p/priyav/PAI_data/all_gis_islandviewer_iv4.csv'
INPUT_SIZE = 22000


islandviewer_file = sys.argv[1]
outfile = sys.argv[2]
print "Input file: ", islandviewer_file
print "Output file: ", outfile
'''
Accesses a particular genomic sequence in NCBI
input: accession number of genome
output: fasta file in NCBI database
'''
def fetch_id(id):
  try:
    handle = Entrez.efetch(db='nucleotide', id=id, rettype='fasta', retmode='text')
    data = SeqIO.read(handle, 'fasta')
    return data.seq
  except urllib2.HTTPError as e:
    print e
    print e.url
  except:
    print id

'''
Gets a set subsequence of the genome
input:
seq_record: sequence output of fetch_id, ie. fetch.seq
start: start of genomic island
stop: ending of genomic island
'''
def get_subsequence(seq_record, start, stop):
  pai = str(seq_record[start:stop])
  return pai

'''
Gets a subsequence of a genome padded by an arbitrary amount of bp such that
the final product is 22kb long
input: sequence, start of desired region, stop of desired region
amount of padding needed to ensure that the final sequence is 22kb
case where start = 1
'''
def get_full_subseq(seq_record, start, stop):
  print 'Getting desired window of sequence...'
  final_size = INPUT_SIZE
  size = (stop-start) + 1 #size of pai region
  pad_size = final_size - size #the amount of total padding needed

  #take some amount of nt from the end and paste onto the front of the sequence
  if start < pad_size:
    new_nt = randint((stop+pad_size), (final_size+pad_size))
    add_to_front = str(seq_record[new_nt:(final_size+pad_size)]) #nt to move to front
    ending = final_size - len(add_to_front) + 1
    rest_of_seq = str(seq_record[start:ending])
    seq = add_to_front + rest_of_seq
  else:
    start_seq = randint((start-pad_size), start) #start the sequence at a random integer before the pai start
    end_pad = final_size - (stop-start_seq)
    end_seq = end_pad+stop
    seq = str(seq_record[start_seq:end_seq])
  print 'Set desired window!'
  return seq

"""
Function to generate random 22kb size fragment from particular genomes
input: genome
output: randomly generated fragment from the input genome
"""
def get_negative_data(seq_record):
  start = randint(0, len(seq_record)-INPUT_SIZE)
  end = start+INPUT_SIZE
  frag = str(seq_record[start:end])
  return frag


acc = [] #list of accession numbers of all sequences
starts = [] #start bp of all sequences
ends = [] #end bp of all sequences
seq_ids = [] #list of accession numbers of 22kb sequences (to be used in data sets)
seqs = [] #list of 22kb sequences (both positve and negative
labels = [] # 1 for PAI, 0 otherwise

"""
use fetch on the stream of ids
and use the indices to get the correct start and stop for each seq
use get_full_subseq on the correct id, start, and stop
grab seqs and put them in the list
write list to csv file
"""

with open(islandviewer_file, 'rb') as file:
  reader = csv.reader(file)
  print 'Reading IslandViewer file...'
  for row in reader:
    acc.append(row[0])
    starts.append(row[1])
    ends.append(row[2])

print 'Finished reading IslandViewer file.'
#remove headers
acc.pop(0)
starts.pop(0)
ends.pop(0)

with open(outfile, 'wb') as csvfile:
  print 'Starting writing data sets to file...'
  filewriter = csv.writer(csvfile, delimiter=",")
  filewriter.writerow(['ID', 'Seq', 'Label'])


  print 'Starting positive data set curation...'
  #get positive dataset
  counter = 0
  for i in range(len(acc)):
    #print 'Sequence number: ' + str(i)
    fetch = fetch_id(acc[i])
    if not fetch:
      print "id or connection to NCBI was faulty", str(i), acc[i]
      continue
    start = int(starts[i])
    end = int(ends[i])
    if end-start <= INPUT_SIZE:
      seq_ids.append(acc[i])
      seq = get_full_subseq(fetch, start, end)
      seqs.append(seq)
      labels.append(1)
      filewriter.writerow([acc[i], seq, '1'])
      counter += 1
    if counter % 500 == 0:
      print "Sequences Processed so far: ", counter
  print 'Finished positive data set curation.'

  ## Find the distribution of lengths of the positive lengths
  lengths = np.array([len(x) for x in seqs])
  print lengths
  plt.hist(lengths, bins=50)
  plt.yscale('log')
  plt.title("Length Distribution of the Positive Dataset")
  plt.savefig('histogram_positive_dataset.png')
  plt.close()

    # Estimate the gaussian distribution of lengths ? is this ok assumption
    #gmm = sklearn.mixture.GaussianMixture(n_components=1)
    #gmm.fit(lengths.reshape(-1, 1))
    #print gmm.means_
    #print gmm.covariances_
    #r = gmm.fit(lengths[:, np.newaxis])
    #print "mean : %f, var : %f" % (gmm.means_[0], gmm.covariances_[0])
    #rv = norm.rvs(loc=gmm_means_[0], scale=np.sqrt(gmm_covariances_[0]))


  print 'Starting negative data set curation...'
  #create negative dataset
  counter = 0
  seq_ids_to_add = []
  for i in range(len(seq_ids)):
    print 'Sequence number: ' + str(i)
    fetch = fetch_id(seq_ids[i])
    seq_ids_to_add.append(seq_ids[i])
    print 'Adding sequence to negative set...'
    seq = get_negative_data(fetch)
    seqs.append(seq)
    labels.append(0)
    filewriter.writerow([seq_ids[i], seq, '0'])
    counter += 1
    if counter % 500 == 0:
      print "Negative Sequences Processed so far: ", counter

print 'Finished negative data set curation.'

#write to new csv file
#with open('database.csv', 'wb') as csvfile:
#  print 'Starting writing data sets to file...'
#  filewriter = csv.writer(csvfile, delimiter=",")
#  filewriter.writerow(['ID', 'Seq', 'Label'])
#  for i in range(len(seq_ids)):
#    filewriter.writerow([seq_ids[i], seqs[i], labels[i]])
#print 'Finished writing data sets to file.'

print 'Finished data curation!'
