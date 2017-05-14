from sklearn import svm
from sklearn import covariance
import numpy as np
import sys
from batchhelper import BatchGenerator

def classify_svm(train_batch_generator, valid_batch_generator, kernel_type):
    """Defines a svm classifier
    
    train_batch_generator: a BatchGenerator with training data
    valid_batch_generator: a BatchGenerator with validation data
    kernel_type: 'linear'|'rbf'|function
    
    Returns:
        correctness rate using that svm with the specified input data
    """
    batch_x, batch_y = train_batch_generator.next_batch()
    flattened_seqs = [np.expand_dims(s.flatten(), axis=0) for s in batch_x]
    b_flattened = np.concatenate(flattened_seqs, axis=0)
    y_flattened = [-1 if x== 0 else 1 for x in batch_y[:,0]]
    
    svc = svm.SVC(kernel=kernel_type, probability=True)

#    batch_x, batch_y = train_batch_generator.next_batch()    
#    b_flattened = np.array([s.flatten() for s in batch_x])
#    y_flattened = [-1 if x== 0 else 1 for x in batch_y[:,0]]

    svc.fit(b_flattened, y_flattened)
    print "finished fitting svm"

    valid_x, valid_y = valid_batch_generator.next_batch()
    b_valid_seqs = [np.expand_dims(s.flatten(), axis=0) for s in valid_x]
    b_valid_flattened = np.concatenate(b_valid_seqs, axis=0)

    y_valid_flattened = [-1 if x== 0 else 1 for x in valid_y[:,0]]
    print "finished valid batch generation"

    #y_pred = svc.predict(b_valid_flattened)
    y_prob = svc.predict_proba(b_valid_flattened)
    #correct_list = [1 if x==y else 0 for x, y in zip(y_valid_flattened, y_pred)]
    print "finished predicting"
    return (svc, np.reshape(np.array(y_valid_flattened), (y_prob.shape[0], 1)),
            y_prob)


# Classification
train_file = ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4aa_data.csv.gz']
valid_file = ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ag_data.csv.gz']
print train_file
print valid_file
train_batch_generator = BatchGenerator(int(sys.argv[2]), train_file, 22000, 1)
valid_batch_generator = BatchGenerator(int(sys.argv[3]), valid_file, 22000, 1)

svc, y_valid_flattened, y_prob = classify_svm(train_batch_generator, valid_batch_generator, 'rbf')

output = np.concatenate([y_valid_flattened, y_prob], axis=1)
np.savetxt(sys.argv[1], output, fmt="%.3f", header='True_Label,%s,%s' % (str(svc.classes_[0]), str(svc.classes_[1])))
