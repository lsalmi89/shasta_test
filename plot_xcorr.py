#!/usr/bin/env python

# Plots ascii xcorr data from first iteration.
# Check that directories are correct and that xcorr.* is correct data type.

import sys
import numpy as np
import matplotlib.pyplot as plt
import filter_types

# Get stations names as input arguments and create filenames for observed and synthetic
dir = '/work/shasta2/lsalmi/SourceStacking.New/Test.Inversion/Iter_0/xcorr/'
st1 = sys.argv[1]
st2 = sys.argv[2]
data_name = np.loadtxt(dir+'xcorr.'+st1+'-'+st2+'.d.dat', delimiter='\n', dtype=None)
syn_name = np.loadtxt(dir+'xcorr.'+st1+'-'+st2+'.g.dat', delimiter='\n', dtype=None)

# Plot observed and synthetic waveforms on same figure
plt.plot(data_name, 'blue', lw = 2, label='Observed')
plt.plot(syn_name, 'green', lw = 2, label='Synthetic')
plt.xlabel('Time',fontname='Arial', fontsize='12', fontweight='bold')
plt.title('cross-correlation '+st1+'-'+st2+' observed vs synthetic',fontname='Arial', fontsize='12', fontweight='bold')
plt.legend(loc=1)
plt.show()

