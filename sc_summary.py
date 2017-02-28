# summarize scan_check info for vdif files

import os
import sys
import commands

files = sys.argv[1:]

for f in files:
    if os.path.getsize(f) <= 20: # skip false recording from Mark6
        continue
    out = commands.getstatusoutput('scan_check -v -c  exthdr= %s' % f)[1].strip().split('\n')
    date = out[0].strip().split()[-1]
    r2head = out[-2][-18:]
    gpsns = out[-1].strip().split()[4]
    passfail = out[1][12:]
    print "%s %s %s %-8s %s" % (date, f, r2head, gpsns, passfail)

