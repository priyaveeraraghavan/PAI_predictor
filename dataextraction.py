from Bio import Entrez
from Bio import SeqIO
from random import randint
import csv

Entrez.email = 'eemartin14@gmail.com'
#islandviewer_dir = '/Users/Liz/Downloads/all_gis_islandviewer_iv4.csv'
islandviewer_dir = '/Users/Sharon/Documents/MIT/Senior\Year/6.802/Final\Project/all_gis_islandviewer_iv4.csv'
INPUT_SIZE = 22000

'''
Accesses a particular genomic sequence in NCBI
input: accession number of genome
output: fasta file in NCBI database
'''
def fetch_id(id):
  handle = Entrez.efetch(db='nucleotide', id=id, rettype='fasta', retmode='text')
  data = SeqIO.read(handle, 'fasta')
  return data.seq

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
  final_size = INPUT_SIZE
  size = (stop-start) + 1 #size of pai region
  pad_size = final_size - size #the amount of total padding needed

  #take some amount of nt from the end and paste onto the front of the sequence
  if start < pad_size:
    new_nt = randint(stop+pad_size, final_size+pad_size)
    add_to_front = str(seq_record[new_nt:final_size+pad_size]) #nt to move to front
    ending = final_size - len(add_to_front) + 1
    rest_of_seq = str(seq_record[start:ending])
    seq = add_to_front + rest_of_seq
  else:
    start_seq = randint(start-pad_size, start) #start the sequence at a random integer before the pai start
    end_pad = final_size - (stop-start_seq)
    end_seq = end_pad+stop
    seq = str(seq_record[start_seq:end_seq])

  return seq


seq_ids = []
starts = []
ends = []
seqs = []
labels = [] # 1 for PAI, 0 otherwise

"""
use fetch on the stream of ids
and use the indices to get the correct start and stop for each seq
use get_full_subseq on the correct id, start, and stop
grab seqs and put them in the list
write list to csv file
"""

#grab sequence ids, starts, and ends of pais from the file
with open(islandviewer_dir, 'rb') as file:
  reader = csv.reader(file)
  for row in reader:
    seq_ids.append(row[0])
    starts.append(row[1])
    ends.append(row[2])

for i in range(len(seq_ids)):
  fetch = fetch_id(seq_ids[i])
  start = starts[i]
  end = ends[i]
  if end-start >= INPUT_SIZE:
    seq = get_full_subseq(fetch, start, end)
    seqs.append(seq)
    labels.append(1)


with open('database.csv', 'wb') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=",")
  filewriter.writerow(['ID', 'Seq', 'Label'])

"""
if __name__ == "__main__":
  with open('all_gis_islandviewer_iv4.csv', 'rb') as file:
    reader = csv.reader(file)
    for row in reader:
      fetch
"""

"""
TODO:

read in csv file correctly
write to new csv file containing id, 22kb seq, positive or negative label

"""
