import argparse
import subprocess
import stat
import os

parser = argparse.ArgumentParser()
parser.add_argument('source', metavar='source', type=str, nargs='+')
parser.add_argument('destination', metavar='destination', type=str)
args = parser.parse_args()

# unmount all disks
for a in open('/proc/mounts'):
    if '/mnt/disks' in a:
        dev = a.split()[0]
        os.system('sudo umount %s' % dev)

# list of sas disks attached
res = subprocess.Popen(["lsscsi", "-t"], stdout=subprocess.PIPE)
out = res.communicate()[0]
disks = [a.split() for a in out.strip().split('\n') if "sas:" in a]

# get physical module number from a lsscsi -t line
# ['[0:0:17:0]', 'disk', 'sas:0x4433221100000000', '/dev/sdai']
def sasmod(disk):
    host, channel, target, lun = map(int, disk[0][1:-1].split(':'))
    mod = 1 + 2*(host > 0) + int(disk[2][-8:-6], 16) / 8
    return mod

# get disk eMSN
diskinfo = dict()
for disk in disks:
    (sasid, dev) = (disk[2], disk[3])
    res = subprocess.Popen(["/usr/bin/sudo", "/sbin/parted", dev, "print"], stdout=subprocess.PIPE)
    out = res.communicate()[0]
    msn = out.strip().split('\n')[-2].strip().split()[5]
    # '[0:0:17:0]' 'disk' 'sas:0x4433221100000000' '/dev/sdai' 1
    # print " ".join(map(repr,disk + [sasmod(disk)]))
    diskinfo[msn] = disk

# mount destination
for i in range(8):
    disk = diskinfo["%s_%d" % (args.destination, i)]
    dev = disk[3]
    m = sasmod(disk)
    os.system('sudo mount %s1 /mnt/disks/%d/%d' % (dev, m, i))

# mount sources read only
for sourcemsn in args.source:
    for i in range(8):
        disk = diskinfo["%s_%d" % (sourcemsn, i)]
        dev = disk[3]
        n = sasmod(disk)
        os.system('sudo mount -o ro %s1 /mnt/disks/%d/%d' % (dev, n, i))

# create copy script
for sourcemsn in args.source:
    for i in range(8):
        disk_in  = diskinfo["%s_%d" % (sourcemsn, i)]
        disk_out = diskinfo["%s_%d" % (args.destination, i)]
        n = sasmod(disk_in)
        m = sasmod(disk_out)
        print "mkdir -p /mnt/disks/%d/%d/%s" % (m, i, sourcemsn)
        print "cp -u /mnt/disks/%d/%d/data/* /mnt/disks/%d/%d/%s/ &" % (n, i, m, i, sourcemsn)
    print "wait"

