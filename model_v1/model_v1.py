from ../training import training

model_type = "basic_CNN"
model_name = "CNN_v1"
model_params = { 'input_shape' : [ 22000, 1, 4],
                 'convolutional_size_params' : [["conv1", [1, 10, 12], 'conv'],
                                                ["pool1", [50, 50, 50, 50], 'pool'],
                                                ["conv2", [1, 10, 9], 'conv'],
                                                ["pool2", [50, 50, 50, 50], 'pool']],
                 'fc_size_params' : [['h_fc1', 20],
                                     ['h_fc2', 10],
                                     ['out', 2]],
                 'l2' : 0.01,
                 'batch_size' : 64,
                 'valid_size' : 3
               }
training_params =  { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                     'valid_size' : 100,
                      'epochs' : 4,
                      'lr' : 0.01, 
                      'train_files' : ['../all_gis_islandviewer_iv4aa_data.csv'],
                      'valid_files' : ['../all_gis_islandviewer_iv4aa_data.csv']}

training(model_name, model_type, model_params, training_params)
