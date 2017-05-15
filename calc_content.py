import numpy as np

data_file = '/Users/Liz/Documents/PAI_Predictor/data/all_gis_islandviewer_iv4ae_data.csv.gz'
all_samples = np.loadtxt(data_file, delimiter=',', skiprows=1, dtype=str)

pos_samps = all_samples[all_samples[:,2] == '1'] #positive samples
neg_samps = all_samples[all_samples[:,2] == '0'] #negative samples

#Calculating GC content and N content for positive samples
pos_gc_content = []
pos_n_content = []
for sample in range(len(pos_samps)):
    gc = 0.0
    ncount = 0.0
    len_seq = float(len(pos_samps[sample,1]))
    for base in pos_samps[sample,1]:
        if base == 'C' or base == 'G':
            gc += 1.0
        elif base == 'N':
            ncount += 1.0
            print ncount
    gc_per = gc / len_seq
    n_per = ncount / len_seq
    pos_gc_content.append(gc_per)
    pos_n_content.append(n_per)

#Calculating average GC content and N content across all positive samples
avg_pos_gc = np.mean(pos_gc_content)
avg_pos_n = np.mean(pos_n_content)
print "Pos GC", avg_pos_gc
print "Pos N", avg_pos_n


#Calculating GC content and N content for negative samples
neg_gc_content = []
neg_n_content = []
for sample in range(len(neg_samps)):
    gc = 0.0
    ncount = 0.0
    len_seq = float(len(neg_samps[sample,1]))
    for base in neg_samps[sample,1]:
        if base == 'C' or base == 'G':
            gc += 1.0
        elif base == 'N':
            ncount += 1.0
    gc_per = gc / len_seq
    n_per = ncount / len_seq
    neg_gc_content.append(gc_per)
    neg_n_content.append(n_per)

#Calculating average GC content and N content across all negative samples
avg_neg_gc = np.mean(neg_gc_content)
avg_neg_n = np.mean(neg_n_content)
#Calculating GC content and N content
pos_gc_content = []
pos_n_content = []
for sample in range(len(pos_samps)):
    gc = 0.0
    ncount = 0.0
    len_seq = float(len(pos_samps[sample,1]))
    for base in pos_samps[sample,1]:
        if base == 'C' or base == 'G':
            gc += 1.0
        elif base == 'N':
            ncount += 1.0
    gc_per = gc / len_seq
    n_per = ncount / len_seq
    pos_gc_content.append(gc_per)
    pos_n_content.append(n_per)

#Calculating average GC content and N content across all samples
avg_pos_gc = np.mean(pos_gc_content)
avg_pos_n = np.mean(pos_n_content)
print "Neg GC", avg_neg_gc
print "Neg N", avg_neg_n