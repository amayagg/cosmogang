"""
Helper file for analyzing GAN performance by plotting loss log file

Authors: Amay Aggarwal, Michel Dellepere, Andrew Ying
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
import pandas as pd 

infile = "output/geogan_master.log"

with open(infile) as f:
    f = f.readlines()

g_losses = []
d_losses = []
for line in f:
	words = line.split()
	for i in range(len(words)):
		if words[i] == "d_loss:":
			d_losses.append(float(words[i+1][:-1]))
		if words[i] == "g_loss:":
			g_losses.append(float(words[i+1][:-1]))

print(len(g_losses), len(d_losses))


g_losses = np.array(g_losses)
d_losses = np.array(d_losses)

#g_losses = g_losses[:400]
#d_losses = d_losses[:400]

#common_factor = 5
#g_losses = np.mean(g_losses.reshape(-1, common_factor), axis=1) # shrink by common_factor
#d_losses = np.mean(d_losses.reshape(-1, common_factor), axis=1)

#g_losses = g_losses[:100]
#d_losses = d_losses[:100]

#print(g_losses)

def moving_average(arr, window):
    """
    http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    """
    ret = np.cumsum(arr)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


#arr = np.arange(20)
g_losses = moving_average(g_losses, window=100)
d_losses = moving_average(d_losses, window=100)
'''
g_losses_df = pd.DataFrame(g_losses.flatten())
d_losses_df = pd.DataFrame(d_losses.flatten())

sns_plot_gen = sns.relplot(kind="line", data=g_losses_df, legend="brief", palette=sns.cubehelix_palette(start=0, n_colors = 1))
sns_plot_gen.savefig("losses/sns_generator_plot.png")

sns_plot_dis = sns.relplot(kind="line", data=d_losses_df, legend="brief", palette=sns.cubehelix_palette(start = 2.8, n_colors = 1))
sns_plot_gen.savefig("losses/sns_discriminator_plot.png")
'''

#g_losses_df = pd.DataFrame(data=g_losses.flatten())
#d_losses_df = pd.DataFrame(data=d_losses.flatten())
'''
sns_plot_gen = sns.relplot(kind="line", data=g_losses_df, legend="brief", palette="ch:2.5, .25")
sns_plot_gen.savefig("losses/sns_generator_plot.png")

sns_plot_dis = sns.relplot( kind="line", data=d_losses_df, legend="brief", palette="ch:1.2, .12")
sns_plot_dis.savefig("losses/sns_discriminator_plot.png")
'''
plt.plot(g_losses, label = "G Loss")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("losses/geogan_gen_loss.png")

plt.figure()
plt.plot(d_losses, label = "D Loss", color = "green")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("losses/geogan_dic_loss.png")
