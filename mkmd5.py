# print md5sum table from first 8192 bytes in each file

import os
import sys
import commands

files = sys.argv[1:]

for f in files:
    if os.path.getsize(f) <= 20: # skip false recording from Mark6
        continue
    out = commands.getstatusoutput('head -c 8192 %s | md5sum' % f)[1][:-3]
    print out + ' ' + f

