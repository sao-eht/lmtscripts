# 1mm localization and total power in dreampy

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
    #           if name[:4] == 'Data')
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
	for i in sorted(scans):
		scan = getscan(i)
		aps.append(detrend(scan.a))
		bps.append(detrend(scan.b))
		ts.append(scan.t + scan.t0)
		xs.append(scan.x)
		ys.append(scan.y)
		ss.append(scan.source)
	t0 = ts[0][0]
	t1 = ts[-1][-1]
	tnew = np.arange(t0, t1+0.2, 0.02)
	idx = np.zeros(len(tnew), dtype=np.bool)
	x = np.zeros(len(tnew))
	y = np.zeros(len(tnew))
	a = np.zeros(len(tnew))
	b = np.zeros(len(tnew))
	s = ss[0]
	for i in range(len(ts)):
		istart = int(np.round((ts[i][0] - t0) * 50.))
		idx[istart:istart+len(ts[i])] = True
		x[istart:istart+len(xs[i])] = xs[i]
		y[istart:istart+len(ys[i])] = ys[i]
		a[istart:istart+len(aps[i])] = aps[i]
		b[istart:istart+len(bps[i])] = bps[i]
	x[~idx] = np.inf
	y[~idx] = np.inf
	return Namespace(t=tnew, a=a, b=b, x=x, y=y, idx=idx, source=s)

def model(x, y, x0=0, y0=0, fwhm=11.):
	fwhm = asec2rad(fwhm)
	sigma = fwhm / 2.335
	# predicted counts
	m = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
	return m

def fitmodel(z, win=50., res=2., fwhm=11.):
	(p, f) = psd(z.b, NFFT=512, pad_to=4096, Fs=50.)
	Fs = 50.
	N = len(z.t)
	np.log2(N)
	pad = 2**int(np.ceil(np.log2(N)))
	fac = np.zeros(pad)
	mpad = np.zeros(pad)
	bpad = np.zeros(pad)
	bpad[:N] = z.b
	B = np.fft.rfft(bpad).conj()
	fm = np.abs(np.fft.fftfreq(pad, d=1./Fs)[:1+pad/2])
	fac = np.median(p) / interp1d(f, p)(fm)
	fac[fm < 0.1] = 0. # turn off low freqs below 0.1 Hz
	x = asec2rad(np.arange(-win, win+res, res))
	y = asec2rad(np.arange(-win, win+res, res))
	(xx, yy) = np.meshgrid(x, y)
	xr = xx.ravel()
	yr = yy.ravel()
	snrs = []
	for (xtest, ytest) in zip(xr, yr):
		mpad[:N] = model(z.x, z.y, xtest, ytest, fwhm=11.)
		M = np.fft.rfft(mpad)
		# take the real part of sum = 0.5 * ifft[0]
		snrs.append(np.sum((M * B * fac).real))
	snr = np.array(snrs)
	snr[snr < 0] = 0.
	imax = np.argmax(snr)
	(xmax, ymax) = (rad2asec(xr[imax]), rad2asec(yr[imax]))
	snr = snr.reshape(xx.shape)
	plt.clf()
	dw = asec2rad(res)
	plt.imshow(snr**2, extent=map(rad2asec, (x[0]-dw/2., x[-1]+dw/2., y[0]-dw/2., y[-1]+dw/2.)), interpolation='nearest', origin='lower')
	plt.ylim(-win, win)
	plt.xlim(-win, win)
	plt.plot(xmax, ymax, 'y+', ms=11, mew=2)
	plt.text(-win, win, '[%.0f, %.0f]' % (xmax, ymax), va='top', ha='left', color='yellow')

def point(first, last=None, win=None, res=2., fwhm=11.):
	if last is None:
		last = first
	scans = range(first, last+1)
	z = mfilt(scans)
	if win is None:
		win = np.ceil(rad2asec(np.abs(np.min(z.x))))
	fitmodel(z, win=win, res=res, fwhm=fwhm)
	if len(scans) == 1:
		plt.title("%s: %d" % (z.source, scans[0]))
	else:
		plt.title("%s: [%d - %d]" % (z.source, scans[0], scans[-1]))


