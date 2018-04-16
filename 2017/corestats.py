# 2015-03-23 LLB remove 1s wait time between snapshots

import corr, adc5g, httplib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sys, time

r2 = corr.katcp_wrapper.FpgaClient('r2dbe-1')
r2.wait_connected()

if len(sys.argv) == 2:
    rpt = int(sys.argv[1])
else:
    rpt = 30

def gaussian(x,a,mu,sig): return a*np.exp(-(x-mu)**2 / (2. * sig**2))

def chisq(par, x, y, yerr):
    (a, mu, sig) = par
    return np.sum((gaussian(x,a,mu,sig)-y)**2/yerr**2)

counts = np.zeros((2,4,256))
x = np.arange(-128, 128, 1)
for r in range(rpt):
    # time.sleep(1)
    x0 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_0_data'))
    x1 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_1_data'))
    for j in range(4):
        bc0 = np.bincount((x0[j::4] + 128))
        bc1 = np.bincount((x1[j::4] + 128))
        counts[0,j,:len(bc0)] += bc0
        counts[1,j,:len(bc1)] += bc1

np.save('counts.npy', counts)

for i in [0,1]:
    for j in [0,1,2,3]:
        y = counts[i,j]
        yerr = np.sqrt(1+y+.10*y**2)
        p0=(np.max(y), 0., 30.)
        ret = scipy.optimize.fmin(chisq, (np.max(y), 0, 40), args=(x, y, yerr), disp=False)
        print "IF%d Core %d: mean %5.2f std %5.2f" % (i, j, ret[1], ret[2])

