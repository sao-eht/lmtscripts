# offset-gain core fitting script using noise
# apt-get install python-scipy
# python og.py clear (clear previous og registers, otherwise all future solutions will be iterative)
# python og.py 3600 (accumulate 3600 snapshots, calculate solution, and apply if setog is True)
# python og.py ogsol-20150320-134300-3600.npy (apply a saved solution)
# 2015.03.20 LLB
# 2015.03.22 LLB remove wait period between snapshot updates

setog = True # whether or not to set registers when training
doplot = True # whether or not to do the matplot png figure

import corr, adc5g, httplib
import numpy as np
import os, sys, time, datetime
try:
    import scipy.optimize
except:
    print "install scipy: apt-get install python-scipy"
    sys.exit()
import warnings
warnings.filterwarnings("ignore")

if doplot:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

corder = [1, 3, 2, 4] # order of cores for spi funcs
counts = None
sol = np.zeros((2,4,2)) # core mean and std fits
x = np.arange(-128, 128, 1) # integer values assigned to bins
xx = np.linspace(-128, 128, 200)

r2 = corr.katcp_wrapper.FpgaClient('r2dbe-1')
r2.wait_connected()

def ogapply(sol):
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            adc5g.set_spi_offset(r2, i, corder[j], sol[i,j,0])
            adc5g.set_spi_gain(r2, i, corder[j], sol[i,j,1])

if len(sys.argv) == 2:
    if os.path.exists(sys.argv[1]) and sys.argv[1][-4:] == ".npy":
        a = np.load(sys.argv[1])
        if a.shape == (2, 4, 2): # is a fit solution
            print "applying solution: %s" % sys.argv[1]
            ogapply(np.load(sys.argv[1]))
            sys.exit()
        else:
            print "must run on fit solution (ogsol-*.npy)"
            sys.exit()
# getting rid of this functionality because it will be wrong if not starting from a clean slate
#         elif a.shape == (2, 4, 256): # is histogram counts
#             print "setting counts from: %s" % sys.argv[1]
#             tag = "-".join(os.path.splitext(os.path.basename(sys.argv[1]))[0].split('-')[1:])
#             counts = a
    elif (sys.argv[1] == "clear") or (sys.argv[1] == "0"):
        print "clearing og registers.."
        ogapply(np.zeros((2,4,2)))
        sys.exit()
    else:
        rpt = int(sys.argv[1])
        tag = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S') + '-%d' % rpt
        print "capturing %d snapshots.." % rpt
else:
    print """
usage:
  python og.py clear (clear previous og registers, otherwise all future solutions will be iterative)
  python og.py 3600 (accumulate 3600 snapshots, calculate solution, and apply if setog is True)
  python og.py ogsol-20150320-134300-3600.npy (apply a saved solution)
"""
    sys.exit()

def gaussian(x,a,mu,sig): return a*np.exp(-(x-mu)**2 / (2. * sig**2))

def chisq(par, x, y, yerr):
    (a, mu, sig) = par
    return np.sum((gaussian(x,a,mu,sig)-y)**2/yerr**2)

if counts is None: # counts not loaded, acquire rpt snapshots
    counts = np.zeros((2,4,256))
    for r in range(rpt): # aggregate counts over multiple snapshots
        # sleep not necessary as adc5g will manually trigger a new snapshot each time it is called
        # time.sleep(1) # wait 1s between grabbing snapshots to get unique
        x0 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_0_data'))
        x1 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_1_data'))
        for j in range(4):
            bc0 = np.bincount((x0[j::4] + 128))
            bc1 = np.bincount((x1[j::4] + 128))
            counts[0,j,:len(bc0)] += bc0
            counts[1,j,:len(bc1)] += bc1
    countsfile = 'ogcounts-%s.npy' % tag
    print "saving counts to: %s" % countsfile
    np.save(countsfile, counts)

# use ADC core counts to do fit and save/apply solution
for i in [0,1]:
    means = np.zeros(4)
    stds = np.zeros(4)
    for j in [0,1,2,3]:
        y = counts[i,j]
        yerr = np.sqrt(1+y+.10*y**2) # 10% systematic error
        p0=(np.max(y), 0., 30.)
        # do fit and ignore first and last bins (saturation)
        ret = scipy.optimize.fmin(chisq, (np.max(y), 0, 40), args=(x[1:-1], y[1:-1], yerr[1:-1]), disp=False)
        if doplot:
            plt.subplot(4,2,1+4*i+j)
            iflabel = 'IF%d core %d' % (i,j)
            statslabel = r'$\mu$:%.1f, $\sigma$:%.1f' % (ret[1], ret[2])
            # h0 = plt.errorbar(x, y, yerr, fmt='.', label='IF%d core %d' % (i,j))
            h0 = plt.plot(x, y, '.', label='IF%d core %d' % (i,j))
            h1 = plt.plot(xx, gaussian(xx, *ret), label=r'$\mu$:%.1f, $\sigma$:%.1f' % (ret[1], ret[2]))
            plt.text(0.05, 0.95, iflabel, ha='left', va='top', transform=plt.gca().transAxes)
            plt.text(0.95, 0.95, statslabel, ha='right', va='top', transform=plt.gca().transAxes)
            plt.xlim(-128, 128)
            plt.ylim(0, 1.05 * np.max(counts))
            plt.yticks([])
            plt.xticks([])
        means[j] = ret[1]
        stds[j] = ret[2]
        print "IF%d Core %d: mean %5.2f std %5.2f" % (i, j, ret[1], ret[2])
    avg_std = np.mean(stds) # target std
    for j in [0,1,2,3]:
        orig_off = adc5g.get_spi_offset(r2, i, corder[j])
        orig_gain = adc5g.get_spi_gain(r2, i, corder[j])
        new_off = orig_off - means[j] * 500./256.
        new_gain = (100. + orig_gain) * (avg_std / stds[j]) - 100.
        if setog:
            adc5g.set_spi_offset(r2, i, corder[j], new_off)
            adc5g.set_spi_gain(r2, i, corder[j], new_gain)
        sol[i,j,0] = new_off
        sol[i,j,1] = new_gain

if doplot:
    plt.suptitle('%s ADC 8bit population\n%s' % (open('/etc/hostname').read().strip(), tag))
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.setp(plt.gcf(), figwidth=8, figheight=12)
    figfile = 'ogplot-%s.png' % tag
    print "saving figure to: %s" % figfile
    plt.savefig(figfile)

solfile = 'ogsol-%s.npy' % tag
print "saving solution to: %s" % solfile
np.save(solfile, sol)
