from Bio import Entrez
from Bio import SeqIO

def fetch_id(id):
  with Entrez.efetch(db='nucleotide', id=id, rettype='fasta', retmode='text') as handle:
    return SeqIO.read(handle, 'fasta')
    
def get_subsequence(seq_record, start, stop):
  return str(seq_record[start:stop])
  
  
