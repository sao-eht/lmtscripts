# summarize scan_check info for vdif files

import os
import sys
import commands

files = sys.argv[1:]

for f in files:
    if os.path.getsize(f) <= 20: # skip false recording from Mark6
        continue
    out = commands.getstatusoutput('scan_check -v -c  exthdr= %s' % f)[1].strip().split('\n')[-2:]
    print "%s %s %s" % (f, out[0][-18:], out[1].strip().split()[4])

