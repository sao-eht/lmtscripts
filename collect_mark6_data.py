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

counts = np.zeros((2,4,256))
x = np.arange(-128, 128, 1)
x0full = np.array([])
x1full = np.array([])
for r in range(rpt):
    # time.sleep(1)
    x0 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_0_data'))
    x1 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_1_data'))
    print x0.shape
    
    x0full.append(x0,axis=0)
    x1full.append(x1,axis=0)
    print x0full.shape


np.save('tmp/dataSamp' + sys.argv[2] + '.npy', x0full, x1full)

