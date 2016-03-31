import subprocess
import os

if os.geteuid() != 0:
    exit("run script as root")

res = subprocess.Popen(["df", "-h"], stdout=subprocess.PIPE)
dfout = res.communicate()[0].split('\n')

minfo = [a for a in open('/proc/mounts') if '/mnt/disks' in a]

# unmount all disks
for a in minfo:
    dev = a.split()[0]
    os.system('umount %s' % dev)

res = subprocess.Popen(["lsscsi", "-t"], stdout=subprocess.PIPE)
out = res.communicate()[0]

drives = []

for line in out.strip().split('\n'):
    tok = line.strip().split()
    if tok[2][:3] == "sas":
        res = subprocess.Popen(["/sbin/parted", tok[3], "print"], stdout=subprocess.PIPE)
        out = res.communicate()[0]
        mod = out.strip().split('\n')[6].strip().split()[5]
        # [17:0:15:0] sas:0x4433221109000000 /dev/sdq EHT%0034_5
        drives.append([tok[0], tok[2], tok[3], mod])

sdrives = sorted(drives, key=lambda x: map(int, x[0][1:-1].split(':')))
mods = []
for i in range(4):
    mods.append(sorted(sdrives[i*8:i*8+8], key=lambda x: x[3]))

# mount read-only
for m in [0,1]:
    for d in range(8):
        os.system('mount -o ro %s1 /mnt/disks/%d/%d' % (mods[m][d][2], m+1, d))
        os.system('mount -o ro %s2 /mnt/disks/.meta/%d/%d' % (mods[m][d][2], m+1, d))

# mount regular
for m in [2,3]:
    for d in range(8):
        os.system('mount %s1 /mnt/disks/%d/%d' % (mods[m][d][2], m+1, d))
        os.system('mount %s2 /mnt/disks/.meta/%d/%d' % (mods[m][d][2], m+1, d))
