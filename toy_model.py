import numpy as np
import architecture as arch

def generate_toy_attention_dataset():
    seqs = []
    for i in range(100):
        motif_length = np.random.randint(5, 10)
        motif = np.ones((motif_length, 4), dtype=np.int8)
        location = np.random.randint(0, 50-motif_length)
        sequence = np.vstack((np.zeros((location, 4), dtype=np.int8), motif, np.zeros((50-location-motif_length, 4), dtype=np.int8)))
        seqs.append(sequence)
    for j in range(100):
        seqs.append(np.vstack([np.array([np.random.randint(0,2) for j in range(4)]) for i in range(50)]))

    X = np.stack(seqs, axis=2)
    X_reshape = np.reshape(X, [200, 50, 4])
    X_expand = np.expand_dims(X_reshape, axis=2)
    return X_expand


X = generate_toy_attention_dataset()
Y = np.reshape(np.hstack((np.vstack((np.ones(100, dtype=np.int8), np.zeros(100, dtype=np.int8))),
               np.vstack((np.zeros(100, dtype=np.int8), np.ones(100, dtype=np.int8))))), (200, 2))

print "Y: [batch_size, num_classes]", Y.shape
print "X: [batch_size, input_length,", X.shape

"""Basic CNN model"""

model_type = "basic_CNN"
model_name = "toy_basic_CNN"
model_params = { 'convolutional_size_params' : [["conv1", [1, 5, 12], 'conv'],
                                                ["pool1", [2, 2, 2, 2], 'pool'],
                                                ["conv2", [7, 7, 9], 'conv'],
                                                ["pool2", [2, 2, 2, 2], 'pool']],
                 'fc_size_params' : [['h_fc1', 20],
                                     ['h_fc2', 10],
                                     ['out', 2]],
                 'l2' : 0.5,
                 'batch_size' : 10,
                 'valid_size' : 3
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                     'valid_size' : 10,
                      'epochs' : 4,
                      'lr' : 0.01}

"""Basic RNN model test"""
model_type = "basic_RNN"
model_name = "toy_basic_RNN"
model_params = { 'unroll_length': 5,
                 'batch_size': 7,
                 'valid_size': 3,
                 'memory_dim': 13,
                 'hidden_fc': 6,
                 'out' : 2,
                 'l2' : 0.5}
                

training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                     'valid_size' : 10,
                      'epochs' : 4,
                      'lr' : 0.01}

# Pick one of the model_params to keep, and one to comment out. 
arch.training(X, Y, X, Y, model_name, model_type, model_params, training_params)
