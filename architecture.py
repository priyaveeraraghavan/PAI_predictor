## Priya Veeraraghavan, Liz Martin, Sharon Wu 2017
## Defines helper functions for PAI predictor architecture construction

class RNN:
    """A RNN model.
        
    Attributes:
        X : the tf placeholder for the input data
        Y : the tf placeholder for the labels 
        unroll_length : the number of characters on which to backprop each time
        batch_size : the size of a batch to process
        cell : GRU cell with size memory_dimension
        initial_state : tf placeholder same size as the GRU cell
        zero_state : zeros the size of the GRU cell, given the batch size. MUST be evaluated in a given session.
        final_states : concatenated output states from each of the windows read by the RNN
        logits : state probabilities following the final states determined by a fully connected output prediction layer.
        fc_layers : a list of tensors [h_fc1, h_fc2 ... logits] 
    """    
    
    def __init__(self, X, Y, size_params, **kwargs):
        """
        
        Arguments:
            X : the tf placeholder for the input data
            Y : the tf placeholder for the labels 
            size_params : dictionary that looks as follows
                { 'unroll_length' : int64,
                  'batch_size' : int64,
                  'memory_dim' : int64,
                  'hidden_fc' : int64,
                  'out_classes' : int64
                  }
                  the input dimension 1 must be divisible by unroll_length for this to work!
        """
          
        self.X = tf.squeeze(X, [2])
        print "X", self.X
        self.Y = Y
        self.unroll_length = size_params['unroll_length']
        self.batch_size = size_params['batch_size']
        self.cell = tf.contrib.rnn.GRUCell(size_params['memory_dim'])
        self.initial_state = tf.placeholder(tf.float32,
                                           [self.batch_size, size_params['memory_dim']],
                                           'initial_state')
        self.zero_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.final_states = self.make_recurrent_graph()
        self.logits = self.make_output_prediction(size_params['hidden_fc'], size_params['out_classes'])
        
        
    def make_recurrent_graph(self):
        
        x_seq_windows = int(self.X.shape[1])/self.unroll_length
        X_inp_list = [tf.split(axis=1, num_or_size_splits=self.unroll_length, value=lst) 
                      for lst in tf.split(axis=1, num_or_size_splits=x_seq_windows, value=self.X)]
        X_inp_list_corrected = [[tf.squeeze(inp_, [1]) for inp_ in x] for x in X_inp_list]
        
        ## simpler version
        #X_inp_list_corrected = [tf.squeeze(inp_, [1]) for inp_ in tf.split(axis=1, num_or_size_splits=int(self.X.shape[1]), value=self.X)]

        print "input list length", len(X_inp_list_corrected)
        print "input list first elt", X_inp_list_corrected[0]
        all_outputs = []
        all_final_states = []
        with tf.variable_scope("rnn") as scope:

            for window in X_inp_list_corrected:
                
                outputs, final_state = tf.contrib.rnn.static_rnn(self.cell, window,
                                   initial_state=self.initial_state)
                all_outputs.append(outputs)
                all_final_states.append(final_state)
                scope.reuse_variables()

        self.final_state = final_state
        
        return tf.concat(axis=1, values=all_final_states)
     

    def make_output_prediction(self, hidden_fc, out):
        print self.final_states.shape
        batch, h = self.final_states.shape
        rnn_out_size = int(h)
        
        flattened_out_states = tf.reshape(self.final_states, [-1, rnn_out_size])
        W1 = tf.Variable(tf.truncated_normal([rnn_out_size, hidden_fc]), name="W1", trainable=True)
        B1 = tf.Variable(tf.zeros([1, hidden_fc]), name="B1", trainable=True)
        hidden1 = tf.nn.tanh(tf.add(tf.matmul(flattened_out_states, W1), B1))
        
        W2 = tf.Variable(tf.truncated_normal([hidden_fc, out]), name="W2", trainable=True)
        B2 = tf.Variable(tf.zeros([out]), name="B2", trainable=True)
        
        out_layer = tf.add(tf.matmul(hidden1, W2), B2)
        self.fc_layers = [hidden1, out_layer]
        return out_layer

  
  class CNN:

    def __init__(self, X, Y, keep_prob, size_params, **kwargs):
        self.X = X
        self.Y = Y
        self.keep_prob = keep_prob
        self.conv_layers, self.pooled_layers, prev_layer= self.make_conv_layers(size_params['convolutional_size_params'])
        self.fc_layers = self.make_fc_layers(size_params['fc_size_params'], prev_layer)
        
    def make_conv_layers(self, convolutional_size_params):
        """ Takes size parameters for an arbitrary number of convolutional layers and returns properly connected conv layers.
    
        Arguments:
            convolutional_size_params: List of the following structure [[layer_name, [filter_height, filter_width, num_filters]]
        
        Returns:
            list of convolutional layers defined with truncated normal weight variables and bias variables initialized to 0.1???
           TODO DECIDE IF constant bias init is correct
        """
        conv_layers = []
        pooled_layers = []
        prev_layer = self.X
        for name, filter_params, pool_or_conv in convolutional_size_params:
            if pool_or_conv == "conv":
                [filter_height, filter_width, num_filters] = filter_params
                print int(prev_layer.shape[2])
                W_conv = tf.Variable(tf.truncated_normal([filter_height, filter_width, int(prev_layer.shape[3]), num_filters], stddev=0.1), name="W" + name)
                b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b" + name)
                h_conv = tf.nn.relu(tf.nn.conv2d(prev_layer, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv, name=name)
                conv_layers.append(h_conv)
                prev_layer = h_conv
            else:
                [filter_size, filter_size, stride, stride] = filter_params
                h_pool = tf.nn.max_pool(h_conv, ksize=[1, filter_size, filter_size, 1],
                        strides=[1, stride, stride, 1], padding='SAME')
                pooled_layers.append(h_pool)
                prev_layer = h_pool
            
        return conv_layers, pooled_layers, prev_layer
    
    def make_fc_layers(self, fc_size_params, prev_layer):
        fc_layers = []
        last_pooled_layer = prev_layer
        _, h, w, d = last_pooled_layer.shape
        print int(h)
        print w
        print d
        prev_out = int(h)*int(w)*int(d)
        prev_layer = tf.reshape(last_pooled_layer, [-1, prev_out])

        for name, out_channel in fc_size_params:           
            W_fc = tf.Variable(tf.truncated_normal([prev_out, out_channel], stddev=0.1), name="W" + name)
            b_fc = tf.Variable(tf.constant(0.1, shape=[out_channel]), name="b" + name)
            h_fc = tf.nn.relu(tf.matmul(prev_layer, W_fc) + b_fc) 
            h_dropout = tf.nn.dropout(h_fc, self.keep_prob) 
            fc_layers.append(h_dropout)
            prev_layer = h_dropout
            prev_out = out_channel
            if name == 'out':
                W_out = tf.Variable(tf.truncated_normal([prev_out, out_channel], stddev=0.1), name="W_out")
                b_out = tf.Variable(tf.constant(0.1, shape=[out_channel]), name="b_out")
                h_out = tf.matmul(prev_layer, W_out) + b_out
                fc_layers.append(h_out)
                break
        
        return fc_layers
        
## ========== FUNCTIONS ==========================================================      
##
## -----------MODEL DEFINITIONS------------
def basic_CNN_model(X, Y, keep_prob, model_params):
    
    cnn = CNN(X, Y, keep_prob, model_params)
    l2_loss = model_params['l2']*sum([tf.nn.l2_loss(x) for x in cnn.fc_layers] + 
                           [tf.nn.l2_loss(y) for y in cnn.conv_layers])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn.fc_layers[-1], labels=Y))
    
    return cnn, loss
  
  
def basic_RNN_model(X, Y, model_params):
    rnn = RNN(X, Y, model_params)
    l2_loss = model_params['l2']*sum([tf.nn.l2_loss(x) for x in rnn.fc_layers])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn.fc_layers[-1], labels=Y))
    #correct_prediction = tf.equal(tf.argmax(rnn.fc_layers[-1],1), tf.argmax(Y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return rnn, loss
  
def random_subset(trX, trY, subset_size):
    subset_idx = list(np.random.choice(len(trX), subset_size))
    subset_X = np.array([trX[idx] for idx in subset_idx])
    subset_Y = np.array([trY[idx] for idx in subset_idx])
    return subset_X, subset_Y

def training(trX, trY, vaX, vaY, model_name, model_type, model_params, training_params):
    """
    Arguments:
        model_name: the name of this model, where all checkpoints will be saved
        model_type: the type of the model, in {'attention_RNN', 'basic_RNN', 'basic_CNN'}
        model_params: dictionary with params for building model. See each model definition
        training_params: dictionary with parameters to train.
            { 'dropout_keep_prob' : float32,
              'max_grad' : float32,
              'valid_size' : int64,
              'epochs' : 3
              }
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
    h, w, d = trX[0].shape
    X = tf.placeholder("float", [None, int(h), int(w), int(d)])
    Y = tf.placeholder("float", [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    
    # Model definition
    # model is a list of tensors
    model_function = {"attention_RNN" : attention_RNN_model,
                      "basic_RNN" : basic_RNN_model,
                      "basic_CNN" : basic_CNN_model}[model_type]
    model, cost = model_function(X, Y, keep_prob, model_params)
    ops = { "basic_RNN" : [model.final_state, cost],
            "basic_CNN" : [cost]}[model_type]

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
    if model_type == "basic_RNN":
        feed_dict[model.initial_state] = model.zero_state.eval(session=sess)
    if model_type == "basic_CNN":
        feed_dict[model.keep_prob] = training_params['dropout_keep_prob']
        

    
    # Start training
    #################################################
    best_cost = float("Inf")
    epochs = training_params['epochs']
    batch_size = model_params['batch_size']
    valid_size = model_params['valid_size']
    for epoch in range(1, epochs+1):
        epoch_cost = 0
        
        # shuffle training data
        trX, trY = random_subset(trX, trY, len(trX))
        
        for batch_idx in range(0, len(trX), batch_size):
            batch_x = trX[batch_idx: (batch_idx + batch_size), :]
            batch_y = trY[batch_idx: (batch_idx + batch_size), :]
            feed_dict[X] = batch_x
            feed_dict[Y] = batch_y
            
            _, batch_cost, state = sess.run([train_op, cost, state],
                                     feed_dict = feed_dict)
            epoch_cost += batch_cost
        
        epoch_cost /= len(range(0, len(trX), batch_size))
        
        # evaluate validation set
        valid_x, valid_y = random_subset(vaX, vaY, valid_size)
        feed_dict[X] = valid_x
        feed_dict[Y] = valid_y
        if model.keep_prob:
            feed_dict[keep_prob] = 1.0
        valid_cost = sess.run(cost, feed_dict = initialization_dict)

        
        # check if this is a superior model
        if valid_cost < best_cost:
            tf.logging.info("Saving best model in %s" % best_model_file)
            saver.save(sess, best_model_file)
            best_cost = valid_cost
            
        # save all models
        saver.save(sess, "".join([model_file, "_epoch", str(epoch), ".ckpt"]))
        tf.logging.info("Epoch: %d - Training: %.3f - Validation %.3f - Best %.3f" % (epoch, epoch_cost, valid_cost, best_cost))
        
    return best_cost
  
def train_RNN(trX, trY, vaX, vaY, model_name, model_type, model_params, training_params):
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
    h, w, d = trX[0].shape
    X = tf.placeholder("float", [None, int(h), int(w), int(d)])
    print "X placeholder", X
    Y = tf.placeholder("float", [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    
    # Model definition
    # model is a list of tensors
    model_function = {"attention_RNN" : attention_RNN_model,
                      "basic_RNN" : basic_RNN_model,
                      "basic_CNN" : basic_CNN_model}[model_type]
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
    feed_dict[model.initial_state] = model.zero_state.eval(session=sess)
    
    # Start training
    #################################################
    best_cost = float("Inf")
    epochs = training_params['epochs']
    batch_size = model_params['batch_size']
    valid_size = model_params['valid_size']
    ops = [cost, model.final_state]
    for epoch in range(1, epochs+1):
        epoch_cost = 0
        
        # shuffle training data
        trX, trY = random_subset(trX, trY, len(trX))
        
        for batch_idx in range(0, len(trX), batch_size):
            batch_x = trX[batch_idx: (batch_idx + batch_size), :]
            batch_y = trY[batch_idx: (batch_idx + batch_size), :]
            if batch_x.shape[0] != batch_size:
                continue
            feed_dict[X] = batch_x
            feed_dict[Y] = batch_y
            batch_cost, state = sess.run(ops,
                                     feed_dict = feed_dict)
            feed_dict[model.initial_state] = state
            epoch_cost += batch_cost

        
        epoch_cost /= len(range(0, len(trX), batch_size))
        
        # evaluate validation set
        valid_x, valid_y = random_subset(vaX, vaY, batch_size)
        feed_dict[X] = valid_x
        feed_dict[Y] = valid_y
        feed_dict[model.initial_state] = model.zero_state.eval(session=sess)
        valid_cost, py = sess.run([cost, model.fc_layers[-1]], feed_dict = feed_dict)
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
  

