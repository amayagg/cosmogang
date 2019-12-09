import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize
import tensorflow as tf
from models import dcgan, utils

checkpoint_dir = 'cosmoGAN_pretrained_weights'
validation_set_location = 'data/cosmogan_maps_256_8k_1.npy' # TODO (team): replace with validation set link
with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:
        gan = dcgan.dcgan(output_size=256,
                          nd_layers=4, # num discrim conv layers
                          ng_layers=4,
                          df_dim=64, # num discrim filters in first conv layer
                          gf_dim=64,
                          z_dim=64, # dimension of NOISE vector z
                          data_format="NHWC")

        gan.inference_graph()

        # TODO (team): modify counter if we are using our own checkpoints ***
        utils.load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, counter=47)

        z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
        samples = sess.run(gan.G, feed_dict={gan.z: z_sample})

        # samples = inverse_transform(samples)
        validation_set = np.load(validation_set_location, mmap_mode='r')
        validation_histogram, bin_edges = np.histogram(validation_set.flatten(), bins=25)
        samples_histogram, _ = np.histogram(samples.flatten(), bins=bin_edges)

        plt.figure()

# def pixel_intensity(validation_set, generator, noise_vector_length, inverse_transform, channel_axis):
#     """First order metric: histogram of pixel intensities
#     Used to compare validation set with the generator set."""
#
#     num = len(validation_set)
#     samples = generator.predict(np.random.normal(size=(num, 1, noise_vector_length)))
