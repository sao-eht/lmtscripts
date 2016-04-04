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

# fs = 50.
# nyq = fs/2.
# Namespace.keys = lambda(self): self.__dict__.keys()

# extract 1mm total power data and fix some timing jitter issues
def extract(nc):
	t0 = nc.hdu.data.Time[0]
	t = nc.hdu.data.Time - t0
	a = nc.hdu.data.APower
	b = nc.hdu.data.BPower
	x = nc.hdu.data.XPos
	y = nc.hdu.data.YPos
	i = ~nc.hdu.data.BufPos.astype(np.bool)
	iobs = nc.hdu.header.ObsNum[0]
	if iobs >= 39150: # move to 50 Hz sampling to avoid ADC time glitches
		fs = 50.
		tnew = nc.hdu.data.Vlbi1mmTpmTime - nc.hdu.data.Vlbi1mmTpmTime[0]
		idx = tnew <= t[-1]
		a = a[idx]
		b = b[idx]
		tnew = tnew[idx]
	elif iobs >= 38983: # kamal includes gap times
		tnew = np.linspace(0, t[-1], len(t))
		fs = 1./(t[1]-t[0])
		if 'Time' in nc.hdu.data['Vlbi1mmTpm']: # use the ADC time if available >= 39118
			adctime = nc.hdu.data.Vlbi1mmTpmTime - nc.hdu.data.Vlbi1mmTpmTime[0]
			tnew = np.linspace(0, adctime[-1], len(adctime))
			tnew = tnew[(tnew <= t[-1])]
			a = interp1d(adctime, a)(tnew)
			b = interp1d(adctime, b)(tnew)
	elif iobs >= 38915: # 83.3 Hz becomes available but has gaps
		fs = 1./0.012
		tnew = np.arange(0, t[-1] + 1e-6, 1./fs)
		a = interp1d(t, a)(tnew) # t is not a great varialbe to use, but all we have
		b = interp1d(t, b)(tnew) # t is not a great varialbe to use, but all we have
	else: # we are in 10 Hz data
		fs = 10.
		tnew = np.arange(0, t[-1] + 1e-6, .10)
		a = interp1d(t, a)(tnew)
		b = interp1d(t, b)(tnew)
	x = interp1d(t, x)(tnew)
	y = interp1d(t, y)(tnew)
	i = interp1d(t, i)(tnew).astype(bool)
	t = tnew
	iobs = nc.hdu.header.ObsNum[0]
	source = nc.hdu.header.SourceName
	return Namespace(t0=t0, t=t, a=a, b=b, x=x, y=y, i=i, iobs=iobs, source=source, fs=fs)

# basic get scan, then extract data from it
def getscan(iobs, do_extract=True):
	from dreampy.onemm.netcdf import OnemmNetCDFFile
	filename = glob('/data_lmt/vlbi1mm/vlbi1mm_*%06d*.nc' % iobs)[-1]
	nc = OnemmNetCDFFile(filename)
	t = nc.hdu.data.Time
	# remove large time glitches
	tmid = np.median(t)
	ibad = np.abs(t-tmid) > 3600
	for i in np.nonzero(ibad)[0]:
		nc.hdu.data.Time[i] = (t[i-1] + t[i+1]) / 2.
	if do_extract:
		return extract(nc)
	else:
		return nc

# raw open (no extract) get original structures
def rawopen(iobs):
	from scipy.io import netcdf
	filename = glob('/data_lmt/vlbi1mm/vlbi1mm_*%06d*.nc' % iobs)[-1]
	nc = netcdf.netcdf_file(filename)
	# keep = dict((name.split('.')[-1], val.data) for (name, val) in nc.variables.items()
	#			if name[:4] == 'Data')
	keep = Namespace()
	keep.BufPos = nc.variables['Data.Dcs.BufPos'].data
	keep.Time = nc.variables['Data.Sky.Time'].data
	keep.XPos = nc.variables['Data.Sky.XPos'].data
	keep.YPos = nc.variables['Data.Sky.YPos'].data
	keep.APower = nc.variables['Data.Vlbi1mmTpm.APower'].data
	keep.BPower = nc.variables['Data.Vlbi1mmTpm.BPower'].data
	if 'Data.Vlbi1mmTpm.Time' in nc.variables:
		keep.ADCTime = nc.variables['Data.Vlbi1mmTpm.Time'].data
	return keep

# export to standard numpy
def exportscan(iobs):
	z = getscan(iobs)
	np.savez('scan_%d' % iobs, **z.__dict__)

# export to standard matlab
def exportmat(iobs):
	z = getscan(iobs)
	scipy.io.savemat('scan_%d.mat' % iobs, z.__dict__)

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
		scan = getscan(i)
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

def model(x, y, x0=0, y0=0, fwhm=11.):
	fwhm = asec2rad(fwhm)
	sigma = fwhm / 2.335
	# predicted counts
	m = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
	return m

def fitmodel(z, win=50., res=2., fwhm=11., channel='b'):
	Fs = z.fs
	tp = z.__dict__[channel]
	# 512 is balance between freq resolution and averaging, good for 50 Hz
	(p, f) = psd(tp, NFFT=512, pad_to=4096, Fs=Fs) # unit variance -> PSD = 1.
	p = p / z.fillfrac # account for zeros in stiched timeseries
	N = len(z.t) # original sequence length
	pad = 2**int(np.ceil(np.log2(N))) # pad length for efficient FFTs
	fac = np.zeros(pad)
	mpad = np.zeros(pad)
	bpad = np.zeros(pad)
	bpad[:N] = tp
	B = np.fft.rfft(bpad).conj() # N factor goes into fft, ifft = 1/N * ..
	fm = np.abs(np.fft.fftfreq(pad, d=1./Fs)[:1+pad/2])
	fac = 1. / interp1d(f, p)(fm) # 1/PSD for matched filter (double whiten)
	fac[fm < 0.1] = 0. # turn off low freqs below 0.1 Hz - just a guess
	x = asec2rad(np.arange(-win, win+res, res))
	y = asec2rad(np.arange(-win, win+res, res))
	(xx, yy) = np.meshgrid(x, y) # search grid
	xr = xx.ravel()
	yr = yy.ravel()
	snrs = [] # signal-to-noise ratios
	norms = [] # sqrt of whitened matched filter signal power
	for (xtest, ytest) in zip(xr, yr):
		mpad[:N] = model(z.x, z.y, xtest, ytest, fwhm=fwhm) # model signal
		M = np.fft.rfft(mpad)
		# take the real part of sum = 0.5 * ifft[0]
		norm = np.sqrt(np.sum(np.abs(M)**2 * fac))
		norms.append(norm)
		snrs.append(np.sum((M * B * fac).real) / norm)
	snr = np.array(snrs)
	snr[snr < 0] = 0.
	imax = np.argmax(snr) # maximum snr location
	(xmax, ymax) = (rad2asec(xr[imax]), rad2asec(yr[imax]))
	snr = snr.reshape(xx.shape)
	plt.clf()
	dw = asec2rad(res)
	plt.imshow(snr**2, extent=map(rad2asec, (x[0]-dw/2., x[-1]+dw/2., y[0]-dw/2., y[-1]+dw/2.)), interpolation='nearest', origin='lower')
	plt.ylim(-win, win)
	plt.xlim(-win, win)
	plt.plot(xmax, ymax, 'y+', ms=11, mew=2)
	plt.text(-win, win, '[%.1f, %.1f]' % (xmax, ymax), va='top', ha='left', color='yellow')
	plt.text(win, win, '[%.2f mV]' % (1e3 * snrs[imax] / norms[imax]), va='top', ha='right', color='yellow')
	print snrs[imax], norms[imax], pad

# (0, 6, 14, 14, 0)
def fitsearch(z, x0=0, y0=0, s10=20., s20=20., th0=0, channel='b'):
	Fs = z.fs
	tp = z.__dict__[channel]
	# 512 is balance between freq resolution and averaging, good for 50 Hz
	(p, f) = psd(tp, NFFT=512, pad_to=4096, Fs=Fs) # unit variance -> PSD = 1.
	p = p / z.fillfrac # account for zeros in stiched timeseries
	N = len(z.t) # original sequence length
	pad = 2**int(np.ceil(np.log2(N))) # pad length for efficient FFTs
	fac = np.zeros(pad)
	mpad = np.zeros(pad)
	bpad = np.zeros(pad)
	bpad[:N] = tp
	B = np.fft.rfft(bpad).conj() # N factor goes into fft, ifft = 1/N * ..
	fm = np.abs(np.fft.fftfreq(pad, d=1./Fs)[:1+pad/2])
	fac = 1. / interp1d(f, p)(fm) # 1/PSD for matched filter (double whiten)
	fac[fm < 0.1] = 0. # turn off low freqs below 0.1 Hz - just a guess
	def snr(args):
		(xtest, ytest, s1test, s2test, thtest) = args
		mpad[:N] = ezmodel(z.x, z.y, xtest, ytest, s1test, s2test, thtest) # model signal
		M = np.fft.rfft(mpad)
		norm = np.sqrt(np.sum(np.abs(M)**2 * fac))
		snr = np.sum((M * B * fac).real) / norm
		return -snr
	result = fmin(snr, (asec2rad(x0), asec2rad(y0), asec2rad(s10)/2.355, asec2rad(s20)/2.355, th0*np.pi/180.))
	print "x: %.1f" % rad2asec(result[0])
	print "y: %.1f" % rad2asec(result[1])
	print "s1: %.2f" % rad2asec(result[2]*2.355)
	print "s2: %.2f" % rad2asec(result[3]*2.355)
	print "th: %.2f" % (result[4] * 180./np.pi)

def fitgrid(z, channel='b'):
	Fs = z.fs
	tp = z.__dict__[channel]
	# 512 is balance between freq resolution and averaging, good for 50 Hz
	(p, f) = psd(tp, NFFT=512, pad_to=4096, Fs=Fs) # unit variance -> PSD = 1.
	p = p / z.fillfrac # account for zeros in stiched timeseries
	N = len(z.t) # original sequence length
	pad = 2**int(np.ceil(np.log2(N))) # pad length for efficient FFTs
	fac = np.zeros(pad)
	mpad = np.zeros(pad)
	bpad = np.zeros(pad)
	bpad[:N] = tp
	B = np.fft.rfft(bpad).conj() # N factor goes into fft, ifft = 1/N * ..
	fm = np.abs(np.fft.fftfreq(pad, d=1./Fs)[:1+pad/2])
	fac = 1. / interp1d(f, p)(fm) # 1/PSD for matched filter (double whiten)
	fac[fm < 0.1] = 0. # turn off low freqs below 0.1 Hz - just a guess
	def makesnr(*args):
		(xtest, ytest, s1test, s2test, thtest) = args
		mpad[:N] = ezmodel(z.x, z.y, xtest, ytest, s1test, s2test, thtest) # model signal
		# mpad[:N] = model(z.x, z.y, xtest, ytest, fwhm=rad2asec(s1test)*2.355)
		M = np.fft.rfft(mpad)
		norm = np.sqrt(np.sum(np.abs(M)**2 * fac))
		snr = np.sum((M * B * fac).real) / norm
		return snr
	(xx, yy, ss1, ss2, tt) = np.mgrid[-2:2, 12:16, 10:30, 10:20, 20:90:15]
	snrs = []
	pars = zip(xx.ravel(), yy.ravel(), ss1.ravel(), ss2.ravel(), tt.ravel())
	for (x, y, s1, s2, th) in pars:
		snrs.append(makesnr(asec2rad(x)/2, asec2rad(y)/2, asec2rad(s1/2.355), asec2rad(s2/2.355), th*np.pi/180.))
	snrs = np.array(snrs)
	ss = snrs.reshape(xx.shape)
	return ss

def point(first, last=None, win=None, res=2., fwhm=11., channel='b'):
	if last is None:
		last = first
	scans = range(first, last+1)
	z = mfilt(scans)
	if win is None:
		win = np.ceil(rad2asec(np.abs(np.min(z.x))))
	fitmodel(z, win=win, res=res, fwhm=fwhm, channel=channel)
	if len(scans) == 1:
		plt.title("%s: %d" % (z.source, scans[0]))
	else:
		plt.title("%s: [%d - %d]" % (z.source, scans[0], scans[-1]))

# general 2D Gaussian
def model2D(x, y, x0, y0, cov11, cov22, cov12):
	invCov = 1.0/(cov11*cov22 - cov12**2) * np.array(( (cov22, -cov12), (-cov12, cov11) ) )
	position = np.array( (x-x0, y-y0) )
	m = np.exp(-0.5 * np.sum(position * np.dot(invCov, position), axis=0))
	return m

def calcCov(sigma1, sigma2, angle):
	vec = np.array( (np.cos(angle),  np.sin(angle) ) ).T; 
	pvec = np.array( (-vec[1], vec[0]) ); 
	eigvals = np.array( ( (sigma1, 0), (0, sigma2) ) )**2
	eigvec = np.array( ( (vec[0], pvec[0]), (vec[1], pvec[1]) ) )
	cov = np.dot(eigvec, np.dot(eigvals, eigvec.T) )
	return cov

def ezmodel(x, y, x0, y0, sigma1, sigma2, angle):
	cov = calcCov(sigma1, sigma2, angle)
	return model2D(x, y, x0, y0, cov[0,0], cov[1,1], cov[1,0])

# def model2dplusring(x, y, x0, y0, cov11, cov22, cov12, ringsize, ringangle, ringrelativeAmplitude, radialprofile):
    
