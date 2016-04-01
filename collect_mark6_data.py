import corr, adc5g, httplib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sys, time

r2 = corr.katcp_wrapper.FpgaClient('r2dbe-1')
r2.wait_connected()

rpt = int(sys.argv[1])

for r in range(rpt):
    # time.sleep(1)
    x0 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_0_data'))
    x1 = np.array(adc5g.get_snapshot(r2, 'r2dbe_snap_8bit_1_data'))
    
    if r > 0:
        x0full = np.column_stack((x0full, x0))
        x1full = np.column_stack((x1full, x1))
    else:
        x0full = x0
        x1full = x1
        
    print x0full.shape


np.save('dataSamp_' + sys.argv[2] + '_x0full.npy', x0full)
np.save('dataSamp_' + sys.argv[2] + '_x1full.npy', x1full)
