# 1mm localization and total power in dreampy
# 2015, 2016 LLB

import numpy
import matplotlib
import shutil
# matplotlib.use('agg')
from matplotlib import pylab, mlab, pyplot
import os
np = numpy
plt = pyplot
# plt.ion()
from argparse import Namespace
from glob import glob
import scipy.io
from scipy.signal import butter,lfilter,freqz
from scipy.interpolate import interp1d
from scipy.ndimage.filters import minimum_filter1d
from scipy.interpolate import UnivariateSpline
from matplotlib.mlab import griddata, psd
from datetime import datetime, timedelta
from scipy.optimize import fmin

def asec2rad(asec):
	return asec * 2*np.pi / 3600. / 360.

def rad2asec(rad):
	return rad * 3600. * 360. / (2*np.pi)

# linear detrend, use only edges
def detrend(x, ntaper=100):
	x0 = np.mean(x[:ntaper])
	x1 = np.mean(x[-ntaper:])
	m = (x1 - x0) / len(x)
	x2 = x - (x0 + m*np.arange(len(x)))
	w = np.hanning(2 * ntaper)
	x2[:ntaper] *= w[:ntaper]
	x2[-ntaper:] *= w[-ntaper:]
	return x2

def model(x, y, x0=0, y0=0, fwhm=11.):
	fwhm = asec2rad(fwhm)
	sigma = fwhm / 2.335
	# predicted counts
	m = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
	return m

# patch together many scans and try to align in time (to the sample -- to keep X and Y)
def mfilt(scans):
	aps = []
	bps = []
	xs = []
	ys = []
	ts = []
	ss = []
	fss = []
	ntaper = 100
	for i in sorted(scans):
		# custom
		scan = Namespace(**scipy.io.loadmat('scandata/scan_%d.mat' % i, squeeze_me=True))
		aps.append(detrend(scan.a, ntaper=ntaper))
		bps.append(detrend(scan.b, ntaper=ntaper))
		ts.append(scan.t + scan.t0)
		xs.append(scan.x)
		ys.append(scan.y)
		ss.append(scan.source)
		fss.append(scan.fs)
	s = ss[0]
	fs = fss[0]
	t0 = ts[0][0]
	t1 = ts[-1][-1]
	tnew = np.arange(t0, t1+1./fs, 1./fs)
	idx = np.zeros(len(tnew), dtype=np.bool)
	x = np.zeros(len(tnew))
	y = np.zeros(len(tnew))
	a = np.zeros(len(tnew))
	b = np.zeros(len(tnew))
	for i in range(len(ts)):
		istart = int(np.round((ts[i][0] - t0) * 50.))
		idx[istart:istart+len(ts[i])] = True
		x[istart:istart+len(xs[i])] = xs[i][:len(x)-istart]
		y[istart:istart+len(ys[i])] = ys[i][:len(y)-istart]
		a[istart:istart+len(aps[i])] = aps[i][:len(a)-istart]
		b[istart:istart+len(bps[i])] = bps[i][:len(b)-istart]
	x[~idx] = np.inf
	y[~idx] = np.inf
	fillfrac = float(np.sum(idx)-ntaper*len(scans)) / len(tnew)
	return Namespace(t=tnew, a=a, b=b, x=x, y=y, idx=idx, source=s, fs=fs, fillfrac=fillfrac)

# custom
def getmat(iobs):
	scan = Namespace(**scipy.io.loadmat('scandata/scan_%d.mat' % iobs, squeeze_me=True))
	return scan

def fitmodel(z, win=50., res=2., fwhm=11., channel='b'):
	Fs = z.fs
	tp = z.__dict__[channel]
	# 512 is balance between freq resolution and averaging, good for 50 Hz
	# (p, f) = psd(tp, NFFT=1024, pad_to=4096, Fs=Fs) # unit variance -> PSD = 1 / Hz
	(p, f) = psd(tp, NFFT=1024, pad_to=4096) # unit variance -> PSD = 1 = variance of complex FFT (1/sqrt(N))
	# we will take out the 1/Hz normalization later, to get unit variance per complex data point
	if 'fillfrac' in z:
		p = p / z.fillfrac # account for zeros in stiched timeseries (otherwise 1)
	N = len(z.t) # original sequence length
	pad = 2**int(np.ceil(np.log2(N))) # pad length for efficient FFTs
	fac = np.zeros(pad)
	mpad = np.zeros(pad)
	bpad = np.zeros(pad)
	ipad = np.zeros(pad).astype(bool)
	# N=32768; xtest=0; ytest=0; win=10.; res=1.; fwhm=11.; channel='b'
	bpad[:N] = tp # fails if N = len(tp) ??
	B = np.fft.rfft(bpad).conj() # N factor goes into fft, ifft = 1/N * ..
	# fm = np.abs(np.fft.fftfreq(pad, d=1./Fs)[:1+pad/2])
	# fac = 1. / interp1d(f, p)(fm) / (Fs/2.) # 1/PSD for matched filter (double whiten), Fs/2 accounts for PSD normalization
	fm = np.abs(np.fft.fftfreq(pad, d=1./2.)[:1+pad/2]) # the default nyquist units
	fac = 1. / interp1d(f, p)(fm) # 1/PSD for matched filter (double whiten), 1. var => 1. fac (no change)
	# apply sqrt(fac) to freq domain normalizes var(x)=1 using any fft/ifft transform pair
	# 1. var => 1. var for 1/sqrt(N) normalization, so x[k] if properly whitened for 1/sqrt(N)
	fac[fm < 0.1 * (2./Fs)] = 0. # turn off low freqs below 0.1 Hz - just a guess
	# np.fft.irfft(B*np.sqrt(fac)) gives unit standard deviation timeseries
	x = asec2rad(np.arange(-win, win+res, res))
	y = asec2rad(np.arange(-win, win+res, res))
	(xx, yy) = np.meshgrid(x, y) # search grid
	xr = xx.ravel()
	yr = yy.ravel()
	snrs = [] # signal-to-noise ratios
	norms = [] # sqrt of whitened matched filter signal power
	for (xtest, ytest) in zip(xr, yr):
		mpad[:N] = model(z.x, z.y, xtest, ytest, fwhm=fwhm) # model signal
		M = np.fft.rfft(mpad) # M big by sqrt(N) factor
		# take the real part of sum = 0.5 * ifft[0]
		norm = np.sqrt(np.sum(np.abs(M)**2 * fac)) # sqrt(N)/sqrt(2) factor total for norm
		norms.append(norm)
		# M=sqrt(N), B=sqrt(N), sum=1/2., norm=sqrt(N)/sqrt(2) => sqrt(N)/sqrt(2) SNR factor
		snrs.append(np.sum((M * B * fac).real) / norm)
	snr = np.array(snrs)
	snr[snr < 0] = 0.
	imax = np.argmax(snr) # maximum snr location
	snr = snr.reshape(xx.shape)
	isnr = np.argsort(snr.ravel())[::-1] # reverse sort high to low
	# snr_true => snr/np.sqrt(pad/2)
	# [snr_true*np.sqrt(pad/2)] / [norm*(sqrt(pad/2))] = htrue
	# snr_true is amplitude of signal in units of normalized filter, need to get in units of unnomralized filter
	prob = np.exp((snr.ravel()/np.sqrt(pad/2.))**2/2.)
	pcum = np.zeros_like(prob)
	pcum[isnr] = np.cumsum(prob[isnr])
	pcum = pcum.reshape(xx.shape) / np.sum(prob)
	xxa = xx * rad2asec(1.)
	yya = yy * rad2asec(1.)
	return Namespace(xx=xxa, yy=yya, snr=snr/np.sqrt(pad/2.), v=1e3*snr/np.array(norms).reshape(xx.shape), prob=prob, pcum=pcum)

def point(first, last=None, win=10., res=0.5, fwhm=11., channel='b', clf=True):
	if last is None:
		last = first
	scans = range(first, last+1)
	z = mfilt(scans)
	if win is None:
		win = np.ceil(rad2asec(np.abs(np.min(z.x))))
	a = fitmodel(z, win=win, res=res, fwhm=fwhm, channel=channel)
	(xxa, yya, snr, v, prob, pcum) = (a.xx, a.yy, a.snr, a.v, a.prob, a.pcum)
	print np.max(snr)
	if clf:
		plt.clf()
	plt.pcolormesh(xxa, yya, 1e3*v)
	# plt.pcolormesh(xxa, yya, s**2)
	plt.colorbar()
	h1 = plt.contour(xxa, yya, pcum, scipy.special.erf(np.array([0,1,2,3])/np.sqrt(2)), colors='blue', lw=2)
	# h1 = plt.contourf(xxa, yya, pcum, scipy.special.erf(np.array([0,1,2,3])/np.sqrt(2)), cmap=plt.cm.get_cmap("Blues"))
	plt.plot(rad2asec(z.x), rad2asec(z.y), 'y-')
	# plt.gca().set_axis_bgcolor('black')
	plt.gca().set_axis_bgcolor('white')
	plt.grid(alpha=0.5)
	plt.ylim(-win, win)
	plt.xlim(-win, win)
	imax = np.argmax(snr.ravel())
	(xmax, ymax) = (xxa.ravel()[imax], yya.ravel()[imax])
	plt.plot(xmax, ymax, 'y+', ms=11, mew=2)
	plt.text(-win, win, '[%.1f, %.1f]' % (xmax, ymax), va='top', ha='left', color='black')
	plt.text(win, win, '[%.2f mV]' % v.ravel()[imax], va='top', ha='right', color='black')
	if len(scans) == 1:
		plt.title("%s: %d" % (z.source, scans[0]))
	else:
		plt.title("%s: [%d - %d]" % (z.source, scans[0], scans[-1]))

def point2(first, last=None, win=10., res=0.5, fwhm=11., channel='b', clf=True):
	if last is None:
		last = first
	scans = range(first, last+1)
	z = mfilt(scans)
	z.x = z.x - np.mean(z.x) # hack
	if win is None:
		win = np.ceil(rad2asec(np.abs(np.min(z.x))))
	a = fitmodel(z, win=win, res=res, fwhm=fwhm, channel=channel)
	(xxa, yya, snr, v, prob, pcum) = (a.xx, a.yy, a.snr, a.v, a.prob, a.pcum)
	n68 = len(pcum.ravel()) - np.sum(pcum.ravel() > 0.68268949213708585)
	a68 = n68 * (res**2)
	e68 = np.sqrt(res**2 + (a68 / np.pi))
	i3s = (pcum.ravel() < 0.99730020393673979)
	v3s = v.ravel()[i3s]
	p3s = prob.ravel()[i3s]
	vmean = np.sum(v3s * p3s) / np.sum(p3s) # expectation value of v3s
	v3s2 = (v3s - vmean)**2
	vstd = np.sqrt(np.sum(v3s2 * p3s) / np.sum(p3s)) # std
	print np.max(snr)
	if clf:
		plt.clf()
	plt.axis(aspect=1.0)
	# plt.pcolormesh(xxa, yya, v, cmap='afmhot_r')
	plt.imshow(v, extent=(-win-res/2., win+res/2., -win-res/2., win+res/2.), interpolation='nearest', origin='lower', cmap='afmhot_r')
	plt.plot(rad2asec(z.x), rad2asec(z.y), '-', color='violet', ls='--', lw=1.5, alpha=0.75)
	# plt.pcolormesh(xxa, yya, s**2)
	# plt.colorbar()
	h1 = plt.contour(xxa, yya, pcum, scipy.special.erf(np.array([0,1,2,3])/np.sqrt(2)), colors='cyan', linewidths=2, alpha=1.0)
	# h1 = plt.contourf(xxa, yya, pcum, scipy.special.erf(np.array([0,1,2,3])/np.sqrt(2)), cmap=plt.cm.get_cmap("Blues"))
	# plt.gca().set_axis_bgcolor('black')
	plt.gca().set_axis_bgcolor('white')
	plt.grid(alpha=0.5)
	plt.ylim(-win-res/2, win+res/2)
	plt.xlim(-win-res/2, win+res/2)
	imax = np.argmax(snr.ravel())
	(xmax, ymax) = (xxa.ravel()[imax], yya.ravel()[imax])
	plt.plot(xmax, ymax, 'y+', ms=11, mew=2)
	plt.text(-0.99*win-res/2, 0.98*win+res/2, '[%.1f, %.1f] $\pm$ %.1f"' % (xmax, ymax, e68), va='top', ha='left', color='black')
	# plt.text(win, win, '[%.2f $\pm$ 2 mV]' % v.ravel()[imax], va='top', ha='right', color='black')
	plt.text(.99*win+res/2, .98*win+res/2, '[%.1f $\pm$ %.1f mV]' % (vmean, vstd), va='top', ha='right', color='black')
	# if len(scans) == 1:
	#	plt.title("%s: %d" % (z.source, scans[0]))
	# else:
	#	plt.title("%s: [%d - %d]" % (z.source, scans[0], scans[-1]))
	plt.title('3C 273')
	plt.xlabel('$\Delta$x [arcsec]')
	plt.ylabel('$\Delta$y [arcsec]')
	plt.gca().set_aspect(1.0)
	plt.tight_layout()

