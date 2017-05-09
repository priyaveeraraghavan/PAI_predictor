import matplotlib
import numpy as np

#this is pretty much ripped straight from pset 1. Need to formulate it for our architecture

def plot_filter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(30,20))
    n_columns = 3
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(np.transpose(units[0,:,:,i]), interpolation="nearest", cmap="afmhot")


# We will get the convolutional layer activations for a specific sample
test_image = test_data[:1]

# h_conv1, h_conv2 are the outputs of your convolutional layers
# ex. h_conv1 = tf.nn.relu(conv2d(x, W) + b)
img_filters1, img_filters2 = sess.run([h_conv1,h_conv2], feed_dict={train_x: test_image})

# Show the original image
plt.imshow(np.transpose(np.reshape(test_image, [x_dim, y_dim])), interpolation="nearest")

# Show the activations of the first convolutional filters for the first test sample
plot_filter(img_filters1)

# Show the activations of the second convolutional filters for the first test sample
#plot_filter(img_filters2)

#Visualization Filter experiment

def getActivations(layer,stimuli):
     units = sess.run(layer,feed_dict={train_x: test_image})
     plot_filter(units)
#getActivations(h_conv1, test_image)
getActivations(h_conv2, test_image)
#plt.save_fig("orig_img1.png", bbox_inches='tight')
