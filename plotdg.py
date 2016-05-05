#!/usr/bin/env python
# Plots observed and synthetic waveforms for three different iterations
# standard summed inversion.

# Need to change the directories.

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('./')
import scipy
import array

# Get stations names as input arguments and create filenames for observed and synthetic
dir = '/work/shasta2/lsalmi/SourceStacking.New/PAVA/Standard/Iter_0/'
dir_iter1 = '/work/shasta2/lsalmi/SourceStacking.New/PAVA/Standard/Iter_2.lower_damp/'
dir_iter2 = '/work/shasta2/lsalmi/SourceStacking.New/PAVA/Standard/Iter_5/'
#dir_iter3 = '/work/shasta2/lsalmi/SourceStacking.New/Test.Inversion/Iter_3/standard/'
st1 = sys.argv[1]

sum_d = dir+'sum.'+st1+'.d'
data_d = array.array('f')
f = open(sum_d, 'rb')
data_d.fromfile(f, 318)

sum_g = dir+'sum.'+st1+'.g'
data_g = array.array('f')
f = open(sum_g, 'rb')
data_g.fromfile(f, 318)

sumiter_g = dir_iter1+'sum.'+st1+'.s'
dataiter_g = array.array('f')
f = open(sumiter_g, 'rb')
dataiter_g.fromfile(f, 318)

sumiter2_g = dir_iter2+'sum.'+st1+'.s'
dataiter2_g = array.array('f')
f = open(sumiter2_g, 'rb')
dataiter2_g.fromfile(f, 318)

#sumiter3_g = dir_iter3+'sum.'+st1+'.g'
#dataiter3_g = array.array('f')
#f = open(sumiter3_g, 'rb')
#dataiter3_g.fromfile(f, 318)

dt = 30
t = dt* np.arange(len(data_d))

fig=plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
# Plot observed and synthetic waveforms on same figure
plt.subplot(3,1,1)
plt.plot(t, data_d, 'blue', lw = 2, label='Observed')
plt.plot(t, data_g, 'green', lw = 2, label='Synthetic')
plt.ylabel('Acceleration (m/s^2)', fontsize='18', fontweight='bold')
plt.yticks(fontsize='16')
plt.xticks(fontsize='16')
plt.title(st1+' observed vs synthetic, 1D prediction', fontsize='18', fontweight='bold')
plt.legend(loc=1)

# Plot observed and synthetic waveforms on same figure for first iteration
plt.subplot(3,1,2)
plt.plot(t, data_d, 'blue', lw = 2, label='Observed')
plt.plot(t, dataiter_g, 'green', lw = 2, label='Synthetic')
plt.ylabel('Acceleration (m/s^2)', fontsize='18', fontweight='bold')
plt.yticks(fontsize='16')
plt.xticks(fontsize='16')
plt.title(st1+' observed vs synthetic, second iteration', fontsize='18', fontweight='bold')
plt.legend(loc=1)

# Plot observed and synthetic waveforms on same figure for second iteration
plt.subplot(3,1,3)
plt.plot(t, data_d, 'blue', lw = 2, label='Observed')
plt.plot(t, dataiter2_g, 'green', lw = 2, label='Synthetic')
plt.ylabel('Acceleration (m/s^2)', fontsize='16', fontweight='bold')
plt.yticks(fontsize='16')
plt.xlabel('Time (s)', fontsize='18', fontweight='bold')
plt.title(st1+' observed vs synthetic, fifth iteration', fontsize='18', fontweight='bold')
plt.legend(loc=1)

# Plot observed and synthetic waveforms on same figure for third iteration
#plt.subplot(4,1,4)
#plt.plot(t, data_d, 'blue', lw = 2, label='Observed')
#plt.plot(t, dataiter3_g, 'green', lw = 2, label='Synthetic')
#plt.xlabel('time (s)', fontsize='12', fontweight='bold')
#plt.title(st1+' observed vs synthetic, third iteration', fontsize='12', fontweight='bold')
#plt.legend(loc=1)

plt.xticks(fontsize='16')
plt.savefig(st1+'_2-5.png' , dpi=120, pad_inches=0)
plt.show()


