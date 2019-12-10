"""
Helper file for analyzing GAN performance by plotting loss log file

Authors: Amay Aggarwal, Michel Dellepere, Andrew Ying
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

infile = "output/cosmo_myExp_batchSize64_flipLabel0.010_nd4_ng4_gfdim64_dfdim64_zdim64_trialgeogan.log"

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

common_factor = 5
g_losses = np.mean(g_losses.reshape(-1, common_factor), axis=1) # shrink by common_factor
d_losses = np.mean(d_losses.reshape(-1, common_factor), axis=1)

#g_losses = g_losses[:100]
#d_losses = d_losses[:100]

print(g_losses)

plt.plot(g_losses, label = "G Loss")
plt.plot(d_losses, label = "D Loss")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("losses/geogan_3.png")
