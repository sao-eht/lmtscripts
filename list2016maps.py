
# list Map scans from saved 1mm data

from scipy.io import netcdf
from glob import glob
from datetime import datetime, timedelta
import numpy as np

files = glob('/data_lmt/vlbi1mm/vlbi1mm_2016-*.nc')
for fname in sorted(files):
	try:
		nc = netcdf.netcdf_file(fname)
		v = nc.variables
		pgm = ''.join(v['Header.Dcs.ObsPgm'].data).strip()
		if pgm == 'Map':
			rate = v['Header.Map.ScanRate'].data * (180/np.pi) * (3600.)
			source = ''.join(v['Header.Source.SourceName'].data).strip()
		elif pgm == 'On':
			rate = 0
			source = 'On'
		else:
			continue
		onum = v['Header.Dcs.ObsNum'].data
		time = v['Data.Sky.Time'].data
		duration = time[-1] - time[0]
		date = str(v['Header.TimePlace.UTDate'].data)
		year = datetime.strptime(date[:4], "%Y")
		fyear = float(date[4:])
		dt = timedelta(days = fyear * 365)
		day = year + dt
		daystr = datetime.strftime(day, "%m/%d %H:%M UT")
		print "%5d  %s  %10s %4.0f %4.0f" % (onum, daystr, source, duration, rate)
	except:
		None
