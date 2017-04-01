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



################### EXTRACT INFORMATION ###################


# extract 1mm total power data and fix some timing jitter issues
def extract_katie(nc):
	t0 = nc.variables['Data.Sky.Time'].data[0]
	t = nc.variables['Data.Sky.Time'].data - t0
	a = nc.variables['Data.Vlbi1mmTpm.APower'].data
	b = nc.variables['Data.Vlbi1mmTpm.BPower'].data
	x = nc.variables['Data.Sky.XPos'].data
	y = nc.variables['Data.Sky.YPos'].data
	i = ~nc.variables['Data.Dcs.BufPos'].data.astype(np.bool)
	iobs = nc.variables['Header.Dcs.ObsNum'].data
	if iobs >= 39150: # move to 50 Hz sampling to avoid ADC time glitches
		fs = 50.
		tnew = nc.variables['Data.Vlbi1mmTpm.Time'].data - nc.variables['Data.Vlbi1mmTpm.Time'].data[0]
		idx = tnew <= t[-1]
		a = a[idx]
		b = b[idx]
		tnew = tnew[idx]
	elif iobs >= 38983: # kamal includes gap times
		tnew = np.linspace(0, t[-1], len(t))
		fs = 1./(t[1]-t[0])
		adctime = nc.variables['Data.Vlbi1mmTpm.Time'].data - nc.variables['Data.Vlbi1mmTpm.Time'].data[0]
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
	#iobs = nc.hdu.header.ObsNum[0]
	source = ''.join(nc.variables['Header.Source.SourceName'])
	return Namespace(t0=t0, t=t, a=a, b=b, x=x, y=y, i=i, iobs=iobs, source=source, fs=fs)

 
def rawopen_katie(iobs):
    from scipy.io import netcdf
    filename = glob('../data_lmt/vlbi1mm/vlbi1mm_*%06d*.nc' % iobs)[-1]
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
    keep.nc = nc
    if 'Data.Vlbi1mmTpm.Time' in nc.variables:
        keep.ADCTime = nc.variables['Data.Vlbi1mmTpm.Time'].data
    return keep
  
# patch together many scans and try to align in time (to the sample -- to keep X and Y)
def mfilt_katie(scans):
    aps = []
    bps = []
    xs = []
    ys = []
    ts = []
    ss = []
    fss = []
    ntaper = 100
    for i in sorted(scans):
        keep = rawopen_katie(i)
        scan = extract_katie(keep.nc)
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

 

################### POINTING ###################


def pointing_lmt2017(first, last=None, plot=True, win=10., res=0.5, fwhm=11., channel='b'):
    
    ############## pointing #############
    
    if last is None:
        last = first
    scans = range(first, last+1)
    z = mfilt_katie(scans)
    if win is None:
        win = np.ceil(rad2asec(np.abs(np.min(z.x))))
        
    # get the prob of a each location in the map being the point source and rename the variables
    out = fitmodel_lmt2017(z, win=win, res=res, fwhm=fwhm, channel=channel)
    (xxa, yya, snr, v, prob, pcum) = (out.xx, out.yy, out.snr, out.v, out.prob, out.pcum)
 
    
    ############## plotting #############
    if plot:
        plt.figure()
        plt.clf()
    
        plt.axis(aspect=1.0)
        plt.imshow(v, extent=(-win-res/2., win+res/2., -win-res/2., win+res/2.), interpolation='nearest', origin='lower', cmap='afmhot_r')
        plt.plot(rad2asec(z.x), rad2asec(z.y), '-', color='violet', ls='--', lw=1.5, alpha=0.75)

        h1 = plt.contour(xxa, yya, pcum, scipy.special.erf(np.array([0,1,2,3])/np.sqrt(2)), colors='cyan', linewidths=2, alpha=1.0)
    
        imax = np.argmax(snr.ravel())
        (xmax, ymax) = (xxa.ravel()[imax], yya.ravel()[imax])
        plt.plot(xmax, ymax, 'y+', ms=11, mew=2)
        plt.text(-0.99*win-res/2, 0.98*win+res/2, '[%.1f, %.1f]"' % (xmax, ymax), va='top', ha='left', color='black')

        plt.title(z.source)
        plt.xlabel('$\Delta$x [arcsec]')
        plt.ylabel('$\Delta$y [arcsec]')
        plt.gca().set_aspect(1.0)
 
        plt.gca().set_axis_bgcolor('white')
        plt.grid(alpha=0.5)
        plt.ylim(-win-res/2, win+res/2)
        plt.xlim(-win-res/2, win+res/2)
    
        plt.tight_layout()
    
    ############## return #############
    
    return out


    
def focusing_lmt2017(first, last=None, plot=True, win=10., res=0.5, fwhm=11., channel='b'):
    
    if last is None:
        last = first       
    vmeans = []
    vstds = []
    z_position = []

    for scan in range(first, last+1):
        
        z_position.append(rawopen_katie(scan).nc.variables['Header.M2.ZAct'].data)

        out = pointing_lmt2017(scan, plot=plot, win=win, res=res, fwhm=fwhm, channel=channel)
        (xxa, yya, snr, v, prob, pcum) = (out.xx, out.yy, out.snr, out.v, out.prob, out.pcum)
     
        i3s = (pcum.ravel() < 0.99730020393673979)
        v3s = v.ravel()[i3s]
        p3s = prob.ravel()[i3s]
        vmean = np.sum(v3s * p3s) / np.sum(p3s) # expectation value of v3s
        v3s2 = (v3s - vmean)**2
        vstd = np.sqrt(np.sum(v3s2 * p3s) / np.sum(p3s)) # std
 
        vmeans.append(vmean)
        vstds.append(vstd)
    
    plt.figure(); plt.errorbar(z_position, vmeans, yerr=vstds)
    
    print np.array(z_position).shape
    print np.array(vmeans).shape

    print np.array(z_position)
    print np.array(vmeans)
    
    vmean_fit = np.polyfit(np.array(z_position), np.array(vmeans), 2)
    p = np.poly1d(vmean_fit)
    znews = np.linspace(np.min(z_position), np.max(z_position),100)
    pnews = p(znews)
    plt.plot(znews, pnews)
    
    #np.polynomial.polynomial.polyfit(x, y, deg, rcond=None, full=False, w=weights)
            
 
    
def focus_mars():
    first = 60709
    last = 60713
    z = mfilt_katie(np.array([first]))
    plt.figure(); plt.plot(z.b)
    
    focusing_lmt2017(first, last=last, plot='False', win=50, channel='b')

    
    
    
    
    
 
def fitmodel_lmt2017(z, win=50., res=2., fwhm=11., channel='b'):
    
    Fs = z.fs
    #extract the detrended voltage measurements
    meas = z.__dict__[channel]

    # original sequence length
    N = len(z.t) 
    # compute pad length for efficient FFTs
    pad = 2**int(np.ceil(np.log2(N))) 
    
    # compute the psd of the voltage measurements
    (p, f) = psd(meas, NFFT=1024, pad_to=pad) # unit variance -> PSD = 1 = variance of complex FFT (1/sqrt(N))
# LINDY COMMENT: we will take out the 1/Hz normalization later, to get unit variance per complex data point
    if 'fillfrac' in z:
        p = p / z.fillfrac # account for zeros in stiched timeseries (otherwise 1)

    fac = np.zeros(pad)
    mpad = np.zeros(pad)
    ipad = np.zeros(pad).astype(bool)
    

    # sample frequencies for a sequence of length 'pad'. This should be equal to f...
    freq_samples = np.abs(np.fft.fftfreq(pad, d=1./2.)[:1+pad/2]) # the default nyquist units
    

    # Compute the factor that whitens the data. This is 1 over the point spread funcntion. 
    # Each of the signals - the model and the measurements - should be whitened by the square root of this term
    whiteningfac_squared = 1. / interp1d(f, p)(freq_samples) # compute 1/PSD at the locations of the measurements B. Really this shouldn't do anything...
    whiteningfac_squared[freq_samples < 0.1 * (2./Fs)] = 0. # turn off low freqs below 0.1 Hz - just an arbitrary choice
    whiteningfac = np.sqrt(whiteningfac_squared)
                 
                 
    # place the measurements into meas_pad so that its padded to be of a power 2 length
    meas_pad = np.zeros(pad)
# lINDY COMMENT: fails if N = len(tp) ??
    meas_pad[:N] = meas 
    
    # measurements of channel volatage in frequency domain
    meas_rfft = np.fft.rfft(meas_pad) # N factor goes into fft, ifft = 1/N * ..
    meas_rfft_conj = meas_rfft.conj(); 
    meas_rfft_conj_white = meas_rfft_conj * whiteningfac


    # compute the x and y coordinates that we are computing the maps over
    x = asec2rad(np.arange(-win, win+res, res))
    y = asec2rad(np.arange(-win, win+res, res))
    (xx, yy) = np.meshgrid(x, y) # search grid
    xr = xx.ravel()
    yr = yy.ravel()
    
    
    snrs = [] # signal-to-noise ratios
    norms = [] # sqrt of whitened matched filter signal power
    for (xtest, ytest) in zip(xr, yr):
        
        # compute the ideal model in the time domain
        mpad[:N] = model(z.x, z.y, xtest, ytest, fwhm=fwhm) # model signal
        
        # convert the ideal model to the frequency domain and whiten
        model_rfft = np.fft.rfft(mpad) 
        model_rfft_white = model_rfft * whiteningfac      

        # compute the normalization by taking the square root of the whitened model spectrums' dot products
        norm = np.sqrt(np.sum(np.abs(model_rfft_white)**2)) 
        norms.append(norm)
        
        snrs.append(np.sum((model_rfft_white * meas_rfft_conj_white).real) / norm)
        
        
    snr = np.array(snrs)
    snr[snr < 0] = 0.
    imax = np.argmax(snr) # maximum snr location
    snr = snr.reshape(xx.shape)
    isnr = np.argsort(snr.ravel())[::-1] # reverse sort high to low
    
    prob = np.exp((snr.ravel()/np.sqrt(pad/2.))**2/2.)
    pcum = np.zeros_like(prob)
    pcum[isnr] = np.cumsum(prob[isnr])
    pcum = pcum.reshape(xx.shape) / np.sum(prob)
    xxa = xx * rad2asec(1.)
    yya = yy * rad2asec(1.)
    
    return Namespace(xx=xxa, yy=yya, snr=snr/np.sqrt(pad/2.), v=1e3*snr/np.array(norms).reshape(xx.shape), prob=prob, pcum=pcum)

 
 
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
