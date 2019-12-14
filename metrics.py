import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from sklearn.decomposition import PCA
import tensorflow as tf
from models import dcgan, utils
from scipy import fftpack
import pandas as pd
from scipy import stats
#import seaborn as sns
#sns.set()
from sklearn.linear_model import LinearRegression

validation_set_location = 'data/dev.npy' # TODO (team): replace with validation set link

def generate_scatter(data):
    #data = np.load("data/dev.npy", mmap_mode = 'r')
    data = tf.reshape(data, [data.shape[0], -1])
    data = data.eval()
    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    pc1 = principalDf['PC1']
    pc2 = principalDf['PC2']
    #print(pc1, pc2)
    #pc1
    '''
    linear_regressor = LinearRegression()
    linear_regressor.fit([pc1], [pc2])
    pc2_pred = linear_regressor.predict([pc1])
    '''
    plt.scatter(pc1, pc2, alpha = 0.5, color = 'g', marker = ".")
    #print(pc1, pc2, pc2_pred)    
    #plt.plot([pc1], pc2_pred, color = 'k')
    plt.title("Baseline PCA")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.axis([-10,10,-10,10])
    plt.savefig("metrics/baseline_pca.png")

#generate_scatter()
def pixel_intensity(centers, histogram):
    plt.figure()
    plt.errorbar(centers, histogram, yerr=np.sqrt(histogram), fmt='v', label='Baseline', color = 'r')
    plt.title('Pixel Intensity Metric')
    plt.xlabel('Pixel Value')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig('metrics/bs_pixint.png', format='png')

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def power_spectrum(image):
    """Computes azimuthal average of 2D power spectrum of a np array image"""
    GLOBAL_MEAN = 0.9998563  # this should be the mean pixel value of the training+validation datasets
    F1 = fftpack.fft2((image - GLOBAL_MEAN) / GLOBAL_MEAN)
    F2 = fftpack.fftshift(F1)
    pspec2d = np.abs(F2) ** 2
    P_k = azimuthalAverage(pspec2d)
    k = np.arange(len(P_k))
    return k, P_k


def batch_Pk(arr):
    """Computes power spectrum for a batch of images"""
    Pk_arr = []
    for idx in range(64):
        k, P_k = power_spectrum(np.squeeze(arr[idx]))
        Pk_arr.append(P_k)
    return k, np.array(Pk_arr)


def pspect(val_imgs, generator, invtransform, noise_vect_len, channel_axis, fname=None, Xterm=True, multichannel=False):
    """plots mean and std deviation of power spectrum over validation set + generated samples"""
    num = val_imgs.shape[0]
    gen_imgs = generator.predict(np.random.normal(size=(num, 1, noise_vect_len)))
    if multichannel:
        gen_imgs = np.take(gen_imgs, 0, axis=channel_axis)  # take the scaled channel
    else:
        gen_imgs = np.squeeze(gen_imgs)
    gen_imgs = invtransform(gen_imgs)
    k, Pk_val = batch_Pk(val_imgs)
    k, Pk_gen = batch_Pk(gen_imgs)

    val_mean = np.mean(Pk_val, axis=0)
    gen_mean = np.mean(Pk_gen, axis=0)
    val_std = np.std(Pk_val, axis=0)
    gen_std = np.std(Pk_gen, axis=0)

    plt.figure()
    plt.fill_between(k, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.4)
    plt.plot(k, gen_mean, 'r--')
    plt.plot(k, val_mean, 'k:')
    plt.plot(k, val_mean + val_std, 'k-')
    plt.plot(k, val_mean - val_std, 'k-')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')
    plt.title('Power Spectrum')
    if Xterm:
        plt.draw()
    else:
        plt.savefig(fname, format='png')
        plt.close()
    return np.sum(np.divide(np.power(gen_mean - val_mean, 2.0), val_mean))

def plot_power_spectrum(validation_images):
    k, Pk_val = batch_Pk(validation_images)
    # k, Pk_gen = batch_Pk(gen_imgs)

    val_mean = np.mean(Pk_val, axis=0)
    # gen_mean = np.mean(Pk_gen, axis=0)
    val_std = np.std(Pk_val, axis=0)
    # gen_std = np.std(Pk_gen, axis=0)

    plt.figure()
    plt.fill_between(k, val_mean - val_std, val_mean + val_std, color='r', alpha=0.25)
    # plt.plot(k, gen_mean, 'r--')
    plt.plot(k, val_mean, 'k:', color = 'black')
    plt.plot(k, val_mean + val_std, 'k-', color = 'black')
    plt.plot(k, val_mean - val_std, 'k-', color = 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')
    plt.title('Baseline Power Spectrum')
    # if Xterm:
    #    plt.draw()
    # else:
    plt.savefig('metrics/bs_psec.png', format='png')
    plt.close()

    #return Pk_val
    #return # np.sum(np.divide(np.power(gen_mean - val_mean, 2.0), val_mean))

def calculate_correlation_matrix(corr_matrix):
    newmat = np.zeros(corr_matrix.shape)
    print(corr_matrix.shape)
    for i in range(corr_matrix.shape[0]-1):
        for j in range(corr_matrix.shape[1]-1):
            newmat[i,j] = corr_matrix[i,j] / np.sqrt(corr_matrix[i,i] * corr_matrix[j,j])
    return newmat

#_, Pk_val = batch_Pk(validation_set)
# _, Pk_gen = batch_Pk(generated_images)
# correlation matrix for validation


#plot_power_spectrum(validation_set)
#plt.matshow(calculate_correlation_matrix(Pk_val), cmap='RdBu', vmin=-1., vmax=1.)
#plt.colorbar()
#plt.savefig("metrics/val_crosscorrnew.png")


with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:
        gan = dcgan.dcgan(output_size=256,
                          nd_layers=4, # num discrim conv layers
                          ng_layers=4,
                          df_dim=64, # num discrim filters in first conv layer
                          gf_dim=64,
                          z_dim=64, # dimension of NOISE vector z
                          transpose_b = False,
                          data_format="NHWC")

        gan.inference_graph()

        # TODO (team): modify counter if we are using our own checkpoints ***
        utils.load_checkpoint(sess, gan.saver, 'dcgan', "checkpoints/cosmoGAN_pretrained_weights", counter = 47)
                
        z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
        samples = sess.run(gan.G, feed_dict={gan.z: z_sample})

        for i in range(3):
            z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
            new_samples = sess.run(gan.G, feed_dict={gan.z: z_sample})
            samples = np.concatenate((samples, new_samples))
        #print(samples.shape)
        samples = samples[:250]
        #samples = np.take(samples, 0, axis=1)
        #print(samples.shape)

        #validation_set = np.load(validation_set_location, mmap_mode='r')
        #print(validation_set.shape)
        plot_power_spectrum(samples)
        #generate_scatter(samples)
        #exit()
        #gen_imgs = np.take(gen_imgs, 0, axis=channel_axis)  # take the scaled channe
        #gen_imgs = np.squeeze(gen_imgs)
        #exit()
        
        #samples = inverse_transform(samples)
        validation_set = np.load(validation_set_location, mmap_mode='r')
        validation_histogram, bin_edges = np.histogram(validation_set.flatten(), bins=25)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        samples_histogram, _ = np.histogram(samples.flatten(), bins=bin_edges)
        #squeezed = np.squeeze(samples, axis = 3)
        print(max(samples_histogram), max(validation_histogram))
        print(stats.ks_2samp(validation_histogram, samples_histogram))
        
        #print(squeezed.shape)
        #print(stats.ks_2samp(validation_set[:64], np.squeeze(samples, axis = 3)))
        #print(validation_set[:64].shape, np.reshape(samples, [64, 256, 256])
        pixel_intensity(centers, samples_histogram)
        exit()
        #print(validation_set.flatten())
        #print(samples.flatten())


