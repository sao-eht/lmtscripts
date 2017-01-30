import subprocess
import stat
import os

showmount = True
showuuid = False

def suid(executable):
    return bool(os.stat(executable).st_mode & stat.S_ISUID)

# /dev/sdr1 /mnt/disks/3/0 xfs rw,relatime,attr2,noquota 0 0
minfo = [a.split() for a in open('/proc/mounts') if '/mnt/disks' in a]
mdict = dict((tok[0],tok[1][-3:]) for tok in minfo)

res = subprocess.Popen(["lsscsi", "-t"], stdout=subprocess.PIPE)
out = res.communicate()[0]
def sassort(tok):
    host, channel, target, lun = map(int, tok[0][1:-1].split(':'))
    return((host, channel, tok[2]))

sas = sorted([a.split() for a in out.strip().split('\n') if "sas:" in a], key=sassort)

for tok in sas:
    (hctl, sasid, dev) = (tok[0], tok[2], tok[3])
    if suid("/sbin/parted"):
        res = subprocess.Popen(["/sbin/parted", dev, "print"], stdout=subprocess.PIPE)
    else:
        res = subprocess.Popen(["/usr/bin/sudo", "/sbin/parted", dev, "print"], stdout=subprocess.PIPE)
    out = res.communicate()[0]
    msn = out.strip().split('\n')[-2].strip().split()[5]
    line = "%-11s %s %-9s %s" % (hctl, sasid, dev, msn)
    # [17:0:15:0] sas:0x4433221109000000 /dev/sdq EHT%0034_5
    if showmount:
        line += "  %s" % mdict.get(dev+"1", "-/-")
    if showuuid:
        res1 = subprocess.Popen(["/sbin/blkid", dev+'1'], stdout=subprocess.PIPE)
        out1 = res1.communicate()[0]
        res2 = subprocess.Popen(["/sbin/blkid", dev+'2'], stdout=subprocess.PIPE)
        out2 = res2.communicate()[0]
        # /dev/sdb1: UUID="f3c22258-7d5a-4be6-b60a-e157b3ac577d" TYPE="xfs""
        uuid1 = out1.strip().split('"')[1]
        uuid2 = out2.strip().split('"')[1]
        line += "  %s %s" % (uuid1, uuid2)
    print line
