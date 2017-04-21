## Priya Veeraraghavan, Liz Martin, Sharon Wu 2017
## Defines helper functions for PAI predictor architecture construction

class CNN:

  def __init__(self, X, Y, convolutional_size_params, **kwargs):
    self.conv_layers = self.make_conv_layers(convolutional_size_params)
    
  def make_conv_layers(self, convolutional_size_params):
    """ Takes size parameters for an arbitrary number of convolutional layers and returns properly connected conv layers.
    
    Arguments:
        convolutional_size_params: List of the following structure [[layer_name, [filter_height, filter_width, num_filters]]
        
    Returns:
        list of convolutional layers defined with truncated normal weight variables and bias variables initialized to 0.1???
       TODO DECIDE IF constant bias init is correct
    """
    conv_layers = []
    prev_layer = X
    for name, filter_params in convolutional_size_params:
      [filter_height, filter_width, num_filters] = filter_params
      W_conv = tf.Variable(tf.truncated_normal([filter_height, filter_weight, prev_layer.shape[3], num_filters], stddev=0.1), name="W" + name)
      b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b" + name)
      h_conv = tf.nn.relu(tf.nn.conv2d(prev_layer, W_conv, strides=[1, stride, stride, 1], padding='SAME') + b_conv, name=name)
      conv_layers.append(h_conv)
      prev_layer = h_conv
    
    return conv_layers
  

class AttentionRNN:
    ##probably don't need to use embedding bc only four amino acids, but may need to change this.
    def __init__(self, X, Y, rnn_params, num_encoder_symbols, num_hidden_units, rnn_params, feed_previous=False, **kwargs):
      """rnn_params : { 'cells' : [hidden_1, hidden_2, ...],
                      'attention' : ?,
                      'prediction_layer' : tanh }
              """
      
      self.context_vector = ?
      self.multi_cell = self.build_cells(rnn_params) 
      self.outputs, self.state = self.build_rnn(....)
      
      def build_cells(self, rnn_params):
        """
        Builds a stack of LSTM cells.
        
        Arguments:
          rnn_params: list of hidden_layer sizes IN ORDER 
        """
        params = {'state_is_tuple': False, 'forget_bias': 1.0}
        cells = []
        for hidden_size in rnn_params.get('cells'):
          cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, **params))
        
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple = False)
        
      
      def build_rnn(self, batch_size, num_attention_heads):
        ## TODO FINISH THIS
        with variable_scope.variable_scope('attention_rnn'):
          # Encoder
          encoder_outputs, encoder_state = core_rnn.static_rnn(self.multi_cell, X) ## may need to slice X into a list of 1d tensors
          
          # concatenate encoder outputs; will put attention on these
          top_states = [array_ops.reshape(o, [-1, 1, self.multi_cell.output_size]) for o in encoder_outputs]
          attention_states = array_ops.concat(top_states, 1)
          
          with variable_scope.variable_scope("attention_decoder", dtype=dtype) as scope:
            batch_size = Y.get_shape()[0].value # array_ops.shape(Y)[0]  # Needed for reshaping.
            attn_length = attention_states.get_shape()[1].value
            if attn_length is None:
              attn_length = array_ops.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value
            hidden_attention = array_ops.reshape(attention_states,
                                                [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            for a in xrange(num_attention_heads):
              k = variable_scope.get_variable("AttnW_%d" % a,
                                              [1, 1, attn_size, attn_size])
              hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
              v.append(variable_scope.get_variable("AttnV_%d" % a [attn_size]))
              ##TODO FINISH ATTENTION
          ##perform some attention alignment
          encoder_state = 
          
          W_pred = tf.Variable(tf.truncated_normal([encoder_state.shape[1], self.params['prediction_layer'][0]], stddev=0.1), name='W_pred')
          b_pred = tf.Variable(tf.constant(0.1, shape=[self.params['prediction_layer][0]]), name="b_pred")
       
          return tf.nn.tanh(tf.add(tf.matmul(encoder_state, W_pred), b_pred))

    
