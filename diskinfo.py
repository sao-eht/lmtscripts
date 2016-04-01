import subprocess
import os

showuuid = False

if os.geteuid() != 0:
    exit("run script as root")

res = subprocess.Popen(["lsscsi", "-t"], stdout=subprocess.PIPE)
out = res.communicate()[0]

for line in out.strip().split('\n'):
    tok = line.strip().split()
    if tok[2][:3] == "sas":
        res = subprocess.Popen(["/sbin/parted", tok[3], "print"], stdout=subprocess.PIPE)
        out = res.communicate()[0]
        mod = out.strip().split('\n')[6].strip().split()[5]
        # [17:0:15:0] sas:0x4433221109000000 /dev/sdq EHT%0034_5
        if showuuid:
            res1 = subprocess.Popen(["/sbin/blkid", tok[3]+'1'], stdout=subprocess.PIPE)
            out1 = res1.communicate()[0]
            res2 = subprocess.Popen(["/sbin/blkid", tok[3]+'2'], stdout=subprocess.PIPE)
            out2 = res2.communicate()[0]
            # /dev/sdb1: UUID="f3c22258-7d5a-4be6-b60a-e157b3ac577d" TYPE="xfs""
            uuid1 = out1.strip().split('"')[1]
            uuid2 = out2.strip().split('"')[1]
            print "%-11s %s %-9s %s %s %s" % (tok[0], tok[2], tok[3], mod, uuid1, uuid2)
        else:
            print "%-11s %s %-9s %s" % (tok[0], tok[2], tok[3], mod)
