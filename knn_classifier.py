from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys
from batchhelper import BatchGenerator

def classify_knn(train_batch_generator, valid_batch_generator, n_neighbors):
    """Defines a knn classifier
    
    train_batch_generator: a BatchGenerator with training data
    valid_batch_generator: a BatchGenerator with validation data
    n_neighbors: number of nearest neighbors
    
    Returns:
        2D array of [predicted classes, real classes, probabilities]  using that knn with the specified input data
    """
    batch_x, batch_y = train_batch_generator.next_batch()
    b_seqs = [np.expand_dims(s.flatten(), axis=0) for s in batch_x]
    b_flattened = np.concatenate(b_seqs, axis=0)
#    b_flattened = np.concatenate([[x, y, z, w] for x, y, z, w in zip(batch_x[0], batch_x[1], batch_x[2], batch_x[3])])
    y_flattened = [-1 if x== 0 else 1 for x in batch_y[:,0]]
    print hi
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

#    batch_x, batch_y = train_batch_generator.next_batch()    
#    b_flattened = np.array([s.flatten() for s in batch_x])
#    y_flattened = [-1 if x== 0 else 1 for x in batch_y[:,0]]

    knn.fit(b_flattened, y_flattened)
    valid_x, valid_y = valid_batch_generator.next_batch()
    b_valid_seqs = [np.expand_dims(s.flatten(), axis=0) for s in valid_x]
    b_valid_flattened = np.concatenate(b_valid_seqs, axis=0)                
    y_valid_flattened = [-1 if x== 0 else 1 for x in valid_y[:,0]]

    y_prob = knn.predict_proba(b_valid_flattened)
    print "finished predicting"
    return (knn, np.reshape(np.array(y_valid_flattened), (y_prob.shape[0], 1)),
            y_prob)



# Classification                                                                                                                           
train_file = ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4aa_data.csv.gz']
valid_file = ['/afs/csail.mit.edu/u/p/priyav/PAI_data/final_data/all_gis_islandviewer_iv4ag_data.csv.gz']
print train_file
print valid_file
train_batch_generator = BatchGenerator(int(sys.argv[2]), train_file, 22000, 1)
valid_batch_generator = BatchGenerator(int(sys.argv[3]), valid_file, 22000, 1)

knn, y_valid_flattened, y_prob = classify_knn(train_batch_generator, valid_batch_generator, 5)


output = np.concatenate([y_valid_flattened, y_prob], axis=1)
np.savetxt(sys.argv[1], output, fmt="%.3f", header='True_Label,%s,%s' % (str(knn.classes_[0]), str(knn.classes_[1])))
