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
#from scipy.ndimage.filters import minimum_filter1dar
from scipy.interpolate import UnivariateSpline
from matplotlib.mlab import griddata, psd
from datetime import datetime, timedelta
from scipy.optimize import fmin

pathname = '../data_lmt/2017/vlbi1mm_*%06d*.nc'
#pathname = '/data_lmt/vlbi1mm/vlbi1mm_*%06d*.nc'

def asec2rad(asec):
	return asec * 2*np.pi / 3600. / 360.

def rad2asec(rad):
	return rad * 3600. * 360. / (2*np.pi)

###################

def focus(first, last, plot=False, point=False, win_pointing=5., win_focusing=5., res=2., fwhm=11., channel='b', z0search=20., alphasearch=20.):
    
    plt.close('all')
    
    if point:
        print 'pointing'
        out = pointing_lmt2017(first, last=last, plot=plot, win=win_pointing, res=res, fwhm=fwhm, channel=channel)
        imax = np.argmax(out.snr.ravel())
        (xmax, ymax) = (out.xx.ravel()[imax], out.yy.ravel()[imax])
    else:
        xmax = 0.
        ymax = 0.

    scans = range(first, last+1)
    print scans
    focus_subset(scans, x0=xmax, y0=ymax, plot=plot, win_pointing=win_pointing, win_focusing=win_focusing, res=res, fwhm=fwhm, channel=channel, z0search=z0search, alphasearch=alphasearch)
    

def focus_subset(scans, x0=0., y0=0., plot=False, win_pointing=50., win_focusing=5., res=2., fwhm=11., channel='b', z0search=20., alphasearch=20.):
    
    focusing_parabolicfit_lmt2017(scans, plot=plot, win=win_pointing, channel=channel)
    focusing_matchfilter_lmt2017(scans, x0=x0, y0=y0, win=win_focusing, res=res, fwhm=fwhm, channel=channel, z0search=z0search, alphasearch=alphasearch)
    
    
    
###########################################################

# Based on scitools meshgrid
def meshgrid_lmtscripts(*xi, **kwargs):
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    .. versionchanged:: 1.9
       1-D and 0-D cases are allowed.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.

        .. versionadded:: 1.7.0
    sparse : bool, optional
        If True a sparse grid is returned in order to conserve memory.
        Default is False.

        .. versionadded:: 1.7.0
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.

        .. versionadded:: 1.7.0

    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.

    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)

    """
    ndim = len(xi)

    copy_ = kwargs.pop('copy', True)
    sparse = kwargs.pop('sparse', False)
    indexing = kwargs.pop('indexing', 'xy')

    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'"
                        % (list(kwargs)[0],))

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1::])
              for i, x in enumerate(xi)]

    shape = [x.size for x in output]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)

################### EXTRACT INFORMATION ###################


# extract 1mm total power data and fix some timing jitter issues
def extract(nc):
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

 
def rawopen(iobs):
    from scipy.io import netcdf
    filename = glob(pathname % iobs)[-1]
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
def mfilt(scans):
    aps = []
    bps = []
    xs = []
    ys = []
    ts = []
    ss = []
    fss = []
    zs = []
    ntaper = 100
    for i in sorted(scans):
        keep = rawopen(i)
        scan = extract(keep.nc)
        aps.append(detrend(scan.a, ntaper=ntaper))
        bps.append(detrend(scan.b, ntaper=ntaper))
        ts.append(scan.t + scan.t0)
        xs.append(scan.x)
        ys.append(scan.y)
        ss.append(scan.source)
        fss.append(scan.fs)
        zs.append(keep.nc.variables['Header.M2.ZReq'].data)
        
    flag = 1
    for s1 in range(0,len(ss)):
        for s2 in range(s1,len(ss)):
            if (ss[s1] != ss[s2]):
                flag = 0
                print('WARNING: NOT THE SAME SOURCE!!')
                print ss
                break
                
            
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
    return Namespace(t=tnew, a=a, b=b, x=x, y=y, z=zs, idx=idx, source=s, fs=fs, fillfrac=fillfrac)

 

################### POINTING & FOCUSING ###################


def pointing_lmt2017(first, last=None, plot=True, win=10., res=0.5, fwhm=11., channel='b', disk_diameter=0.):
    
    if last is None:
        last = first
    scans = range(first, last+1)
    
    out = pointing_lmt2017_wrapper(scans, plot=plot, win=win, res=res, fwhm=fwhm, channel=channel, disk_diameter=disk_diameter)
    
    return out

def pointing_lmt2017_wrapper(scans, plot=True, win=10., res=0.5, fwhm=11., channel='b', disk_diameter=0.):
    
    ############## pointing #############

    z = mfilt(scans)
    if win is None:
        win = np.ceil(rad2asec(np.abs(np.min(z.x))))
        
    # get the prob of a each location in the map being the point source and rename the variables
    out = fitmodel_lmt2017(z, win=win, res=res, fwhm=fwhm, channel=channel, disk_diameter=disk_diameter)
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

        plt.title(z.source.strip() + '     scans:' + str(scans[0]) + '-' + str(scans[-1]))
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

        
def focusing_parabolicfit_lmt2017(scans, plot=True, win=10., res=0.5, fwhm=11., channel='b'):

          
    vmeans = []
    vstds = []
    z_position = []

    for scan in scans:
        
        z_position.append(rawopen(scan).nc.variables['Header.M2.ZReq'].data)

        out = pointing_lmt2017(scan, plot=plot, win=win, res=res, fwhm=fwhm, channel=channel)
        (xxa, yya, snr, v, prob, cumulative_prob) = (out.xx, out.yy, out.snr, out.v, out.prob, out.pcum)

# KATIE: DOESN'T THIS ONLY WORK IF YOU ONLY HAVE 1 PEAK???

        # get the indices of the points on the map within 3 sigma and extract those voltages and probablities
        indices_3sigma = (cumulative_prob.ravel() < 0.99730020393673979)
        voltages_3sigma = v.ravel()[indices_3sigma]
        prob_3sigma = prob.ravel()[indices_3sigma]

        # compute the expected value of the source voltage within 3 sigma
        sourcevoltage_expvalue = np.sum(voltages_3sigma * prob_3sigma) / np.sum(prob_3sigma) # expectation value of v3s
        
        # compute the variance of the source voltage within 3 sigma
        voltage_squareddiff = (voltages_3sigma - sourcevoltage_expvalue)**2
        sourcevoltage_variance = np.sqrt(np.sum(voltage_squareddiff * prob_3sigma) / np.sum(prob_3sigma)) # std
 
        vmeans.append(sourcevoltage_expvalue)
        vstds.append(sourcevoltage_variance)
    
    plt.figure(); plt.errorbar(z_position, vmeans, yerr=vstds)
    
    ############ LEAST SQUARES FITTING ################
    
    
    A = np.vstack([np.ones([1, len(z_position)]), np.array(z_position), np.array(z_position)**2]).T
    meas = np.array(vmeans)
    meas_cov = np.diag(np.array(vstds)**2)
    
     
    polydeg = 2
    scale = 1e5
    polyparams_cov = scale*np.eye(polydeg+1)
    polyparams_mean = np.zeros([polydeg+1])
    
    intTerm = np.linalg.inv(meas_cov + np.dot(A, np.dot(polyparams_cov, A.T)))
    est_polyparams = polyparams_mean + np.dot(polyparams_cov, np.dot(A.T, np.dot( intTerm, (meas - np.dot(A, polyparams_mean)) ) ) ) 
    error_polyparams = polyparams_cov  - np.dot(polyparams_cov, np.dot(A.T, np.dot(intTerm, np.dot(A, polyparams_cov)) ) )
    
    #print 'estimated polyparams'
    #print est_polyparams
    #print 'estimated error'
    #print error_polyparams
    
    p = np.poly1d(est_polyparams[::-1])
    znews = np.linspace(np.min(z_position), np.max(z_position),100)
    pnews = p(znews)
    plt.plot(znews, pnews)    
    
    imax = np.argmax(pnews)
    z0 = znews[imax]

    print 'estimated z0'
    print z0

    ##################################################
    
    #vmean_fit_flipped, stats = np.polynomial.polynomial.polyfit(np.array(z_position), np.array(vmeans), 2, rcond=None, full=True, w=1/np.array(vstds))
    #vmean_fit = vmean_fit_flipped[::-1]
    
    #p = np.poly1d(vmean_fit)
    #znews = np.linspace(np.min(z_position), np.max(z_position),100)
    #pnews = p(znews)
    #plt.plot(znews, pnews)   


    #plt.text(-1.4, 210., '[estimated $\mathbf{z}_0$: %.3f $\pm$ %.3f]' % (z0, z0_approxstdev), va='top', ha='left', color='black')
    plt.text(min(znews), min(pnews), '[peak $\mathbf{z}$: %.3f]' % (z0), va='top', ha='left', color='black')

        
    plt.title('Focusing')
    plt.xlabel('$\mathbf{z}$')
    plt.ylabel('amplitude')     

    
    
def focusing_matchfilter_lmt2017(scans, x0=0, y0=0, win=50., res=2., fwhm=11., channel='b', alpha_min=0., alpha_max=20., disk_diameter=0., z0search=20., alphasearch=20., plot=True):
    
    print 'warning! recomputing N for each scan'
    
    all_scans = mfilt(scans)
    if win is None:
        win = np.ceil(rad2asec(np.abs(np.min(all_scans.x))))
    
    zpos = []
    xpos = []
    ypos = []
    meas_whitened = []
    for scan_num in scans:
        
        scan = mfilt(range(scan_num,scan_num+1))
        
        # place the measurements into meas_pad so that its padded to be of a power 2 length
        meas = scan.__dict__[channel]

        # original sequence length
        N = len(scan.t) 
        # compute pad length for efficient FFTs
        pad = 2**int(np.ceil(np.log2(N))) 
        
        if scan_num == scans[0]:
            whiteningfac = whiten_measurements(all_scans, pad, channel=channel)
    
        meas_pad = np.zeros(pad)
        meas_pad[:N] = meas 
    
        # measurements of channel volatage in frequency domain
        meas_rfft = np.fft.rfft(meas_pad) # N factor goes into fft, ifft = 1/N * ..
        meas_rfft_conj = meas_rfft.conj(); 
        meas_rfft_conj_white = meas_rfft_conj * whiteningfac 
        
        meas_whitened.append(meas_rfft_conj_white)
        zpos.append(scan.z[0])
        xpos.append(scan.x)
        ypos.append(scan.y)
    
    z0_min = min(zpos)
    z0_max = max(zpos)
    z0s = np.linspace(z0_min, z0_max, z0search)
    alphas = np.linspace(alpha_min, alpha_max,alphasearch)    
    
    # compute the x and y coordinates that we are computing the maps over
    x = asec2rad(np.arange(x0-win, x0+win+res, res))
    y = asec2rad(np.arange(y0-win, y0+win+res, res))
    
    #(z0s_grid, alphas_grid, xx_grid, yy_grid) = np.meshgrid(z0s, alphas, x, y) # search grid
    (z0s_grid, alphas_grid, xx_grid, yy_grid) = meshgrid_lmtscripts(z0s, alphas, x, y) # search grid
    zr = z0s_grid.ravel()
    ar = alphas_grid.ravel()
    xr = xx_grid.ravel()
    yr = yy_grid.ravel()
    
    count = 0.
    num_zs = len(zpos)
    model_pad = np.zeros(pad)
    snrs = [] # signal-to-noise ratios
    norms = [] # sqrt of whitened matched filter signal power
    for (ztest, atest, xtest, ytest) in zip(zr, ar, xr, yr):
        
        #print count/len(zr)
        
        if disk_diameter > 0:
            models = focus_model_disk(xpos, ypos, zpos, x0=xtest, y0=ytest, fwhm=fwhm, z0=ztest, alpha=atest, disk_diameter=disk_diameter, res=0.2)
        else:
            models = focus_model(xpos, ypos, zpos, x0=xtest, y0=ytest, fwhm=fwhm, z0=ztest, alpha=atest)
        
        
        snr =  0.0
        norm = 0.0
        for s in range(0,num_zs):
            
            N = len(models[s])
            
            # compute the ideal model in the time domain
            model_pad[:N] = models[s]
        
            # convert the ideal model to the frequency domain and whiten
            model_rfft = np.fft.rfft(model_pad) 
            model_rfft_white = model_rfft * whiteningfac      

            # compute the normalization by taking the square root of the whitened model spectrums' dot products
            norm = norm + np.sum(np.abs(model_rfft_white)**2)
            snr = snr + ( np.sum((model_rfft_white * meas_whitened[s]).real) )
        
        norm = np.sqrt(norm)
        norms.append(norm)
        snrs.append(snr/norm)
        count = count + 1.
        
    # compute probablity and cumulative probabilities
    isnr = np.argsort(np.array(snrs).ravel())[::-1] # reverse sort high to low
    prob = np.exp((np.array(snrs).ravel()/np.sqrt(num_zs * pad/2.))**2/2.)
    pcum = np.zeros_like(prob)
    pcum[isnr] = np.cumsum(prob[isnr])
    pcum = pcum.reshape(z0s_grid.shape) / np.sum(prob)
    
    # get the indices of the points on the map within 3 sigma and extract those z0s and probablities
    indices_3sigma = (pcum.ravel() < 0.99730020393673979)
    z0s_3sigma = z0s_grid.ravel()[indices_3sigma]
    prob_3sigma = prob.ravel()[indices_3sigma]
    # compute the expected value of the z0 within 3 sigma
    z0_expvalue = np.sum(z0s_3sigma * prob_3sigma) / np.sum(prob_3sigma) # expectation value of v3s
    # compute the variance of the source voltage within 3 sigma
    z0_squareddiff = (z0s_3sigma - z0_expvalue)**2
    z0_variance = np.sqrt(np.sum(z0_squareddiff * prob_3sigma) / np.sum(prob_3sigma)) # std



    imax = np.argmax(np.array(snrs).ravel())
    (zmax, amax, xmax, ymax) = (zr.ravel()[imax], ar.ravel()[imax], xr.ravel()[imax], yr.ravel()[imax])
        
    print 'estimated z0'
    print zmax
    
        
    if plot:
        
        plt.figure()
        plt.clf()
        
        loc = np.unravel_index(imax, xx_grid.shape)
        reshape_snr = np.array(snrs).reshape(z0s_grid.shape)
        slice_snr = reshape_snr[:,:,loc[2],loc[3]]
        
        plt.imshow(slice_snr, extent=(z0_min, z0_max, alpha_min, alpha_max), aspect=(z0_max-z0_min)/(alpha_max-alpha_min), interpolation='nearest', origin='lower', cmap='Spectral_r')
        h1 = plt.contour(z0s_grid[:,:,loc[2],loc[3]], alphas_grid[:,:,loc[2],loc[3]], pcum[:,:,loc[2],loc[3]], scipy.special.erf(np.array([0,1,2,3])/np.sqrt(2)), colors='cyan', linewidths=2, alpha=1.0)
                
        
        plt.plot(zmax, amax, 'y+', ms=11, mew=2)
        plt.text(z0_min+z0s[1]-z0s[0], alpha_max-(alphas[1]-alphas[0]), '[maximum $\mathbf{z}_0$: %.3f,  x: %.3f,  y: %.3f, alpha: %.3f]' % (zmax, rad2asec(xmax), rad2asec(ymax), amax), va='top', ha='left', color='black')
        plt.text(z0_min+z0s[1]-z0s[0], alpha_max-4*(alphas[1]-alphas[0]), '[expected $\mathbf{z}_0$: %.3f $\pm$ %.3f]' % (z0_expvalue, np.sqrt(z0_variance)), va='top', ha='left', color='black')

        
        plt.title('Focusing')
        plt.xlabel('$\mathbf{z}_0$')
        plt.ylabel('alpha (FWHM in arcseconds per mm offset in $\mathbf{z}$)')
 
        plt.gca().set_axis_bgcolor('white')    
        plt.tight_layout()

    return
    

    
def whiten_measurements(z, pad_psd, channel='b'):
    
    Fs = z.fs
    #extract the detrended voltage measurements
    meas = z.__dict__[channel]


    # compute the psd of the voltage measurements
    (p, f) = psd(meas, NFFT=1024, pad_to=4096) # unit variance -> PSD = 1 = variance of complex FFT (1/sqrt(N))
# LINDY COMMENT: we will take out the 1/Hz normalization later, to get unit variance per complex data point
    if 'fillfrac' in z:
        p = p / z.fillfrac # account for zeros in stiched timeseries (otherwise 1)

    # sample frequencies for a sequence of length 'pad'. This should be equal to f...
    freq_samples = np.abs(np.fft.fftfreq(pad_psd, d=1./2.)[:1+pad_psd/2]) # the default nyquist units
    
    # Compute the factor that whitens the data. This is 1 over the point spread funcntion. 
    # Each of the signals - the model and the measurements - should be whitened by the square root of this term
    whiteningfac_squared = 1. / interp1d(f, p)(freq_samples) # compute 1/PSD at the locations of the measurements B. Really this shouldn't do anything...
    whiteningfac_squared[freq_samples < 0.1 * (2./Fs)] = 0. # turn off low freqs below 0.1 Hz - just an arbitrary choice
    whiteningfac = np.sqrt(whiteningfac_squared)
    
    return whiteningfac
    

def fitmodel_lmt2017(z, win=50., res=2., fwhm=11., channel='b', disk_diameter=0.):
    
    Fs = z.fs
    #extract the detrended voltage measurements
    meas = z.__dict__[channel]

    # original sequence length
    N = len(z.t) 
    # compute pad length for efficient FFTs
    pad = 2**int(np.ceil(np.log2(N))) 
    
    whiteningfac = whiten_measurements(z, pad, channel=channel)
    
    # place the measurements into meas_pad so that its padded to be of a power 2 length
    modelpad = np.zeros(pad)
    meas_pad = np.zeros(pad)
    meas_pad[:N] = meas  # lINDY COMMENT: fails if N = len(tp) ??
    
    # measurements of channel volatage in frequency domain
    meas_rfft = np.fft.rfft(meas_pad) # N factor goes into fft, ifft = 1/N * ..
    meas_rfft_conj = meas_rfft.conj(); 
    meas_rfft_conj_white = meas_rfft_conj * whiteningfac 
    
    # compute the x and y coordinates that we are computing the maps over
    x = asec2rad(np.arange(-win, win+res, res))
    y = asec2rad(np.arange(-win, win+res, res))
    #(xx, yy) = np.meshgrid(x, y) # search grid
    (xx, yy) = meshgrid_lmtscripts(x, y) # search grid
    xr = xx.ravel()
    yr = yy.ravel()
    
    
    count = 0; 
    snrs = [] # signal-to-noise ratios
    norms = [] # sqrt of whitened matched filter signal power
    for (xtest, ytest) in zip(xr, yr):
        
        # compute the ideal model in the time domain
        if disk_diameter>0:
            modelpad[:N] = model_disk(z.x, z.y, x0=xtest, y0=ytest, fwhm=fwhm, disk_diameter=disk_diameter, res=disk_diameter/8.)
        else:
            modelpad[:N] = model(z.x, z.y, x0=xtest, y0=ytest, fwhm=fwhm) # model signal
        
        # convert the ideal model to the frequency domain and whiten
        model_rfft = np.fft.rfft(modelpad) 
        model_rfft_white = model_rfft * whiteningfac      

        # compute the normalization by taking the square root of the whitened model spectrums' dot products
        norm = np.sqrt(np.sum(np.abs(model_rfft_white)**2)) 
        norms.append(norm)
        
        snrs.append(np.sum((model_rfft_white * meas_rfft_conj_white).real) / norm)
        
        count = count + 1
        
        
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
    
    # m = model, b = measurements, 
    # Expected [ b_conj * (noise + amplitude*m) ] 
    #          = Expected [b_conj*noise + b_conj*amplitude*m] = 0 + amplitude*b_conj*m
    # Optimally, m = b. Therefore to get out the amplitude we would need to divide by
    # b_conj*m = |model|^2 = norms^2
    volts2milivolts = 1e3
    voltage = volts2milivolts * snr/ np.array(norms).reshape(xx.shape)
    
    return Namespace(xx=xxa, yy=yya, snr=snr/np.sqrt(pad/2.), v=voltage, prob=prob, pcum=pcum)

   

 
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



def focus_model(xpos, ypos, zs, x0=0, y0=0, fwhm=11., z0=0, alpha=0):
    
    fwhm2stdev_factor = 1/2.335
    
    sigma = asec2rad(fwhm) * fwhm2stdev_factor
    alpha_rad = asec2rad(alpha) * fwhm2stdev_factor
      
    count = 0
    models = []
    for z in zs:
        sigma_z = np.sqrt(sigma**2 + (alpha_rad*np.abs(z-z0))**2)
        amplitude_z = 1/( np.sqrt(2*np.pi) * (sigma_z)**2 )
        m_z = amplitude_z * np.exp(-((xpos[count]-x0)**2 + (ypos[count]-y0)**2) / (2*sigma_z**2))
        models.append(m_z)
        count = count + 1
        
    return models


def model_disk(xpos, ypos, x0=0, y0=0, fwhm=11., disk_diameter=0., res=2.):
    
    fwhm2stdev_factor = 1/2.335
    
    sigma = asec2rad(fwhm) * fwhm2stdev_factor 
    res_rad = asec2rad(res)       
    sigma_pixels = sigma/res_rad

    # generate a disk image at radian positions xx and yy
    disk_radius = asec2rad(disk_diameter)/2.
    
    
    x = (np.arange(x0-disk_radius-3*sigma, x0+disk_radius+3*sigma+res_rad, res_rad))
    y = (np.arange(y0-disk_radius-3*sigma, y0+disk_radius+3*sigma+res_rad, res_rad))
    
    #(xx_disk, yy_disk) = np.meshgrid(x, y) # search grid
    (xx_disk, yy_disk) = meshgrid_lmtscripts(x, y) # search grid
    disk = np.zeros(xx_disk.shape)
    disk[ ((xx_disk-x0 )**2 + (yy_disk- y0)**2) <= disk_radius**2 ] = 1.
    
    blurred_disk = scipy.ndimage.filters.gaussian_filter(disk, sigma_pixels, mode='constant', cval=0.0)

    #interpfunc = scipy.interpolate.RectBivariateSpline(xx_disk.flatten(), yy_disk.flatten(), blurred_disk.flatten())
    interpfunc = scipy.interpolate.RectBivariateSpline(y, x, blurred_disk)
    model = interpfunc(ypos, xpos,  grid=False)
    
    return model

    
def focus_model_disk(xpos, ypos, zs, x0=0, y0=0, fwhm=11., z0=0, alpha=0, disk_diameter=0., res=2.):
    
    # generate a disk image at radian positions xx and yy
    disk_radius = asec2rad(disk_diameter)/2.
    scaling = 10
    res_rad = asec2rad(res)  
    x = (np.arange(x0-scaling*disk_radius, x0+scaling*disk_radius+res_rad, res_rad))
    y = (np.arange(y0-scaling*disk_radius, y0+scaling*disk_radius+res_rad, res_rad))
    #(xx_disk, yy_disk) = np.meshgrid(x, y) # search grid
    (xx_disk, yy_disk) = meshgrid_lmtscripts(x, y) # search grid
    disk = np.zeros(xx_disk.shape)
    disk[ ((xx_disk-x0)**2 + (yy_disk-y0)**2) <= (disk_radius)**2 ] = 1.
    
    
    fwhm2stdev_factor = 1/2.335
    
    sigma = asec2rad(fwhm) * fwhm2stdev_factor
    alpha_rad = asec2rad(alpha) * fwhm2stdev_factor
      
    count = 0
    models = []
    for z in zs:
        sigma_z = np.sqrt(sigma**2 + (alpha_rad*np.abs(z-z0))**2)
        amplitude_z = 1/( np.sqrt(2*np.pi) * (sigma_z)**2 )
        
        sigma_z_pixels = sigma_z/asec2rad(res)
        blurred_disk = scipy.ndimage.filters.gaussian_filter(disk, sigma_z_pixels, mode='constant', cval=0.0)

        #interpfunc = scipy.interpolate.RectBivariateSpline(xx_disk.flatten(), yy_disk.flatten(), blurred_disk.flatten())
        interpfunc = scipy.interpolate.RectBivariateSpline(y, x, blurred_disk)
        m_z = interpfunc(ypos[count], xpos[count],  grid=False)

        models.append(m_z)
        count = count + 1
    
    #plt.figure(); plt.imshow(blurred_disk)
    return models

