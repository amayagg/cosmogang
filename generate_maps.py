"""
This file generates convergence maps from a GAN with pre-trained weights.

Authors: Amay Aggarwal, Michel Dellepere, Andrew Ying
"""

import tensorflow as tf
from models import dcgan, utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

checkpoint_dir = 'cosmoGAN_pretrained_weights'

with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:
        gan = dcgan.dcgan(output_size=256,
                      nd_layers=4,
                      ng_layers=4,
                      df_dim=64,
                      gf_dim=64,
                      z_dim=64,
                      data_format="NHWC")
        
        gan.inference_graph()
        
        utils.load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, counter=47)
        
        z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
        samples = sess.run(gan.G, feed_dict={gan.z: z_sample})
        
        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        # g_vars = [var for var in t_vars if 'generator/' in var.name]
        
        # g_vars = [(sess.run(var), var.name) for var in g_vars]# if 'g_h' in var.name]
        # d_vars = [(sess.run(var), var.name) for var in d_vars]#if 'd_h' in var.name]

norm = pltcolors.LogNorm(1e-4, samples[5].max(), clip='True')
plt.imsave('cmap.png', np.squeeze(samples[5]), norm=norm, cmap=plt.get_cmap('Blues'))
# plt.imshow(np.squeeze(samples[5]), norm=norm, cmap=plt.get_cmap('Blues'));
