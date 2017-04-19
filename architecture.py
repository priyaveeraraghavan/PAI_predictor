## Priya Veeraraghavan, Liz Martin, Sharon Wu 2017
## Defines helper functions for PAI predictor architecture construction

class ConvLayers:

  def __init__(self, convolutional_size_params, **kwargs):
    self.conv_layers = self.make_conv_layers(convolutional_size_params)
    
  def make_conv_layers(self, convolutional_size_params):
    """ Takes size parameters for an arbitrary number of convolutional layers and returns properly connected conv layers.
    
    Arguments:
        convolutional_size_params: List of the following structure [[layer_name, [filter_height, filter_width, num_filters]]
        
    Returns:
        list of convolutional layers defined with truncated normal weight variables and bias variables initialized to 0.
    
    TODO: implement
    """
    pass
    
 class AttentionNetwork:
  
    def __init__(self, 
    
    
