# print md5sum table from first 8192 bytes in each file

import sys
import commands

files = sys.argv[1:]

for f in files:
    out = commands.getstatusoutput('head -c 8192 %s | md5sum' % f)[1][:-3]
    print out + ' ' + f

