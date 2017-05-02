from batchhelper import BatchGenerator
from architecture import basic_RNN_model
from architecture import basic_CNN_model
import os
import tensorflow as tf
def training(model_name, model_type, model_params, training_params):
    """Trains a model given the specifications.
    
    Arguments:
      model_name: string name of model
      model_type : string, "basic_RNN" or "basic_CNN"
      model_params : dictionary of parameters, depending on model_type --
        CNN ----------
            { 'input_shape' : [ 22000, 1, 4],
                 'convolutional_size_params' : [["conv1", [1, 10, 12], 'conv'],
                                                ["pool1", [50, 50, 50, 50], 'pool'],
                                                ["conv2", [1, 10, 9], 'conv'],
                                                ["pool2", [50, 50, 50, 50], 'pool']],
                 'fc_size_params' : [['h_fc1', 20],
                                     ['h_fc2', 10],
                                     ['out', 2]],
                 'l2' : 0.01,
                 'batch_size' : 10,
                 'valid_size' : 3
               }
        RNN ----------
            { 'input_shape' : [ 22000, 1, 4],
                 'unroll_length': 5,
                 'batch_size': 7,
                 'valid_size': 3,
                 'memory_dim': 13,
                 'hidden_fc': 6,
                 'out' : 2,
                 'l2' : 0.5}
       training_params : dictionary of training parameters:
            { 'dropout_keep_prob' : 1.0,
                     'max_grad' : 0.01,
                     'valid_size' : 10,
                      'epochs' : 4,
                      'lr' : 0.01, 
                      'train_files' : ['first_ten_out.csv', 'first_ten_out.csv'],
                      'valid_files' : ['first_ten_out.csv']}
    """
    # File and directory specifications
    model_dir = os.path.join(".", model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    model_file = os.path.join(model_dir, model_name)
    best_model_file = os.path.join(model_dir, "".join([model_name, "_best.ckpt"]))
    
    # Define graph
    tf.reset_default_graph()
    
    # Input and output
    #X = {...}[model_type]
    h, w, d = model_params['input_shape']
    X = tf.placeholder("float", [None, int(h), int(w), int(d)])
    print "X placeholder", X
    Y = tf.placeholder("float", [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    
    # Model definition
    # model is a list of tensors
    model_function = {"basic_RNN" : basic_RNN_model,
                      "basic_CNN" : basic_CNN_model}[model_type]
    if "RNN" in model_type:
        rnn = True
    else:
        rnn = False
        
    model, cost = model_function(X, Y, model_params)

    # Training Procedure
    global_step = tf.get_variable('global_step', [],
                              initializer=tf.constant_initializer(0.0))
    train_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, train_vars), training_params['max_grad'])
    optimizer = tf.train.RMSPropOptimizer(training_params['lr'], 0.9)
    train_op = optimizer.apply_gradients(zip(grads, train_vars),
                                     global_step=global_step)
    
    # Initialize variables
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=1000000)
    init = tf.global_variables_initializer()
    sess.run(init)
    print "Initializing variables..."
    
    feed_dict = {}
    if rnn:
        feed_dict[model.initial_state] = model.zero_state.eval(session=sess)

        
    # check validation before any training
    #valid_size = training_params['valid_size']
    #valid_X, valid_Y = random_subset(vaX, vaY, valid_size)
    #_, best_cost = sess.run([train_op, cost], feed_dict = { X: valid_X,
                                                            #Y: valid_Y,
                                                           # keep_prob: 1.0 })
    
    # Start training
    #################################################
    best_cost = float("Inf")
    epochs = training_params['epochs']
    batch_size = model_params['batch_size']
    trainfiles = training_params['train_files']
    train_batch_generator = BatchGenerator(batch_size, trainfiles)
    valid_size = model_params['valid_size']
    validfiles = training_params['valid_files']
    valid_batch_generator = BatchGenerator(valid_size, validfiles)
    
    ops = [cost]
    if rnn:
        ops += [model.final_state]
    
    
    for epoch in range(1, epochs+1):
        epoch_cost = 0
        
        # shuffle training data
        # trX, trY = random_subset(trX, trY, len(trX))
        train_batch_generator.reset()
        
        #for batch_idx in range(0, len(trX), batch_size):
            #batch_x = trX[batch_idx: (batch_idx + batch_size), :]
            #batch_y = trY[batch_idx: (batch_idx + batch_size), :]
        iterations = 0
        while True:
            try:
                batch_x, batch_y = train_batch_generator.next_batch()
                print "running next batch..."
                #if batch_x.shape[0] != batch_size:
                #    continue
                feed_dict[X] = batch_x
                feed_dict[Y] = batch_y
                feed_dict[model.keep_prob] = training_params['dropout_keep_prob']
            
                if rnn:
                    batch_cost, state = sess.run(ops,
                                     feed_dict = feed_dict)
                    feed_dict[model.initial_state] = state
                else:
                    [batch_cost] = sess.run(ops, feed_dict = feed_dict)
                
                epoch_cost += batch_cost
                iterations += 1
            except StopIteration:
                print "end of epoch!"
                break

        
        #epoch_cost /= len(range(0, len(trX), batch_size))
        epoch_cost /= iterations
        
        # evaluate validation set
        #valid_x, valid_y = random_subset(vaX, vaY, batch_size)
        #feed_dict[X] = valid_x
        #feed_dict[Y] = valid_y
        valid_batch_generator.reset()
        valid_x, valid_y = valid_batch_generator.next_batch()
        feed_dict[model.keep_prob] = 1.0
        if rnn:
            feed_dict[model.initial_state] = model.zero_state.eval(session=sess)
        valid_cost, py = sess.run([cost, model.classification_py], feed_dict = feed_dict)
        print py

        
        # check if this is a superior model
        if valid_cost < best_cost:
            tf.logging.info("Saving best model in %s" % best_model_file)
            saver.save(sess, best_model_file)
            best_cost = valid_cost
            
        # save all models
        saver.save(sess, "".join([model_file, "_epoch", str(epoch), ".ckpt"]))
        tf.logging.info("Epoch: %d - Training: %.3f - Validation %.3f - Best %.3f" % (epoch, epoch_cost, valid_cost, best_cost))
        
    return best_cost
