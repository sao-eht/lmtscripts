#!/usr/bin/env python

# missing tests:
#   DONE: vv time offset (needs root)
#   DONE: NTP offset is small
#   DONE: all 8 disks are present
#   DONE: disks are not too full (rtime?)
#   DONE: modules are in open state ready to record (dplane status? should work)
#   DONE: input stream is sensible
#   DONE: GPS PPS offset is not unreasonably large (can happen if GPS unlocked)
#   DONE: modules are in a group? - assume this is covered by the other tests
#   DONE: (dplane state check) group=open was done before input_stream ? (i.e. input streams armable)
#   DONE: mark6 interrupts in /etc/default/mark6

# vv test is impossible right now:
# oper@Mark6-4047:~$ sudo /home/oper/bin/vv -i eth3 -p 4001 -n 5
# [sudo] password for oper:
# Sorry, user oper is not allowed to execute '/home/oper/bin/vv -i eth3 -p 4001 -n 5' as root on localhost.
# oper@Mark6-4047:~$ /home/oper/bin/vv -i eth3 -p 4001 -n 5
# eth3: You don't have permission to capture on that device (socket: Operation not permitted)

import sys
import unittest
import corr
import httplib
import os
import re
import numpy as np
import adc5g
import time
import subprocess
import socket
from datetime import datetime, timedelta
from pkg_resources import parse_version

if len(sys.argv) == 1:
    sys.argv.append('-v')

def get_switch_ip():
    code = open("/usr/local/src/r2dbe/software/switch_set_IF.py").read()
    return code.split('"')[1]

r2_hostname = 'r2dbe-1'
input_streams = {'eth3':'12', 'eth5':'34'} # input streams we will test for
switch_ip = get_switch_ip()
m6_software_version = '1.2j'
r2_bitcode_md5sum = '6421249e83aa86a9f2630b2c2ea04d22'
if_power_tol = 10 # deviation of std from ideal in ADC 8bit units
r2_threshold_tol = 4 # deviation of th from ideal in ADC 8bit units
vv_threshold = 0.05 # seconds offset after which to warn for vv packet vs system time
ntp_threshold = 0.05 # seconds offset after which to warn about NTP offset size
gpspps_threshold = 1e-5 + (5e-12 * 86400 * 20) # reasonable PPS drift since lock + GPS scatter

# socket defn to cplane
socket_res = socket.getaddrinfo('127.0.0.1', 14242, socket.AF_INET, socket.SOCK_STREAM)[0]
def cplanecmd(cmd):
    af, socktype, proto, canonname, sa = socket_res
    s = socket.socket(af,socktype, proto)
    s.connect(sa)
    s.sendall(cmd + ';') # extra ';' will not matter
    ret = s.recv(8192).strip()
    s.close()
    return ret

roach2 = corr.katcp_wrapper.FpgaClient(r2_hostname)
roach2.wait_connected()
x0 = np.array(adc5g.get_snapshot(roach2, 'r2dbe_snap_8bit_0_data'))
x1 = np.array(adc5g.get_snapshot(roach2, 'r2dbe_snap_8bit_1_data'))
th0 = roach2.read_int('r2dbe_quantize_0_thresh')
th1 = roach2.read_int('r2dbe_quantize_1_thresh')

class R2EpochIsCorrect(unittest.TestCase):
    def test(self):
        utcnow = datetime.utcnow()
        epoch = 2*(utcnow.year - 2000) + (utcnow.month > 6)
        self.assertEqual(epoch, roach2.read_int('r2dbe_vdif_0_hdr_w1_ref_ep'), "epoch set in R2DBE startup script")
        self.assertEqual(epoch, roach2.read_int('r2dbe_vdif_1_hdr_w1_ref_ep'), "epoch set in R2DBE startup script")

# class R2SecondsAreCorrect(unittest.TestCase):
#     def test(self):
#         utcnow = datetime.utcnow()
#         wait = (1500000 - utcnow.microsecond) % 1e6 # get to 0.5s boundary
#         time.sleep(wait / 1e6)
#         utcnow = datetime.utcnow()
#         refdate = datetime(utcnow.year, 1+6*(utcnow.month > 6), 1)
#         # print refdate
#         dt = utcnow - refdate
#         totalsec = dt.days * 86400 + dt.seconds
#         gpscnt = roach2.read_uint('r2dbe_onepps_gps_pps_cnt')
#         r2sec0 = gpscnt + roach2.read_int('r2dbe_vdif_0_hdr_w0_sec_ref_ep')
#         r2sec1 = gpscnt + roach2.read_int('r2dbe_vdif_1_hdr_w0_sec_ref_ep')
#         # print r2sec0 - totalsec
#         self.assertEqual(totalsec, r2sec0)
#         self.assertEqual(totalsec, r2sec1)

class R2BitcodeIsUpToDate(unittest.TestCase):
    def test(self):
        import hashlib
        path = '/srv/roach2_boot/current/boffiles/r2dbe_rev2.bof'
        md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
        self.assertEqual(md5, r2_bitcode_md5sum, "R2DBE bitcode in git repository")

class R2IsConnected(unittest.TestCase):
    def test(self):
        self.assertTrue(roach2.is_connected(), "Check ethernet connection and MAC addresses in Mark6 dnsmasq")

class R2GpsIncrementedBy1(unittest.TestCase):
    def test(self):
        utcnow = datetime.utcnow()
        wait = (1500000 - utcnow.microsecond) % 1e6 # get to 0.5s boundary
        time.sleep(wait / 1e6)
        gpspps1 = roach2.read_uint('r2dbe_onepps_gps_pps_cnt')
        time.sleep(1.0)
        gpspps2 = roach2.read_uint('r2dbe_onepps_gps_pps_cnt')
        self.assertTrue(gpspps2 == gpspps1 + 1, "Check GPS PPS signal")

class R2ClockIs256MHz(unittest.TestCase):
    def test(self):
        a=roach2.read_uint('sys_clkcounter')
        b=roach2.read_uint('sys_clkcounter')
        time.sleep(0.1)
        c=roach2.read_uint('sys_clkcounter')
        time.sleep(1.0)
        d=roach2.read_uint('sys_clkcounter')
        clk = (c-2*b+a)/1e5
        # temporary extra diagnostic for this test
        self.assertTrue(abs(clk - 256.) < 2.0, "Check 10 MHz and 2048 clock synth [%.2f] %d %d %d %d" % (clk, a, b, c, d))

class IFPowerIsGood(unittest.TestCase):
    def test(self):
        self.assertTrue(np.abs(np.std(x0) - 30.) <= if_power_tol and
                np.abs(np.std(x1) - 30.) <= if_power_tol, "Verify IF power and BDC attenuators")

class IFThresholdIsGood(unittest.TestCase):
    def test(self):
        self.assertTrue(np.abs(np.std(x0) - th0) <= r2_threshold_tol and
                np.abs(np.std(x1) - th1) <= r2_threshold_tol, "Run alc.py")

class SwitchIsSetToIF(unittest.TestCase):
    def test(self):
        connection = httplib.HTTPConnection(switch_ip,80)
        connection.request("GET","/SWPORT?")
        response = connection.getresponse()
        data = response.read()
        self.assertEqual(int(data), 0, "Run switch_set_IF.py")

class Mark6SoftwareIsCurrent(unittest.TestCase):
    def test(self):
        self.assertTrue(os.path.exists('/usr/local/src/Mark6_%s' % m6_software_version), "nstall new Mark6 software")

class TimezoneIsUTC(unittest.TestCase):
    def test(self):
        self.assertEqual(time.tzname[1], 'UTC', "Mark6 must be set to UTC")

class Mark6InterruptsCorrect(unittest.TestCase):
    def test(self):
        line = (l for l in open('/etc/default/mark6') if l[:8] == 'MK6_OPTS').next()
        self.assertTrue(all(dev in line for dev in input_streams.keys()), "/etc/default/mark6 does not match input_streams defined at top")

class NTPOffsetIsSmall(unittest.TestCase):
    def test(self):
        res = subprocess.Popen("ntpq -pn".split(), stdout=subprocess.PIPE)
        offsetms = float(res.communicate()[0].strip().split('\n')[-1].split()[-2])
        self.assertTrue(abs(offsetms) < 1e3*ntp_threshold, "Try to restart NTP and monitor offset")

class GPSPPSOffsetIsSmall(unittest.TestCase):
    def test(self):
        noffset = roach2.read_int('r2dbe_onepps_offset')
        self.assertTrue(abs(noffset) < gpspps_threshold * 256.e6, "GPS-vs-PPS offset is large, was the GPS lock good?")

class SASDisksAreAllThere(unittest.TestCase):
    def test(self):
        res = subprocess.Popen("lsscsi -t".split(), stdout=subprocess.PIPE)
        ndisk = res.communicate()[0].count("sas:")
        self.assertTrue(ndisk == 32, "Found %d SAS disks not 32" % ndisk)

class SASDisksAreAllMounted(unittest.TestCase):
    def test(self):
        nmount = open('/proc/mounts').read().count('/mnt/disks/')
        self.assertTrue(nmount == 64, "Found %d mount points not 32 (data+meta)")

class EnoughSpaceFor10hrs(unittest.TestCase):
    def test(self):
        rtime = cplanecmd('rtime?32000;') # 32000 Mbps
        rtimesec = float(rtime.split(':')[4])
        self.assertTrue(rtimesec >= 10*3600, "only %.1f hours left on modules!" % (rtimesec/3600.))

class cplaneIsRunningAndUnder1GB(unittest.TestCase):
    def test(self):
        res = subprocess.Popen("/bin/ps ho rss -C cplane".split(), stdout=subprocess.PIPE)
        out = res.communicate()[0]
        self.assertTrue(out != "" and int(out) < 1e6, "Run /etc/init.d/cplane restart")

class dplaneIsRunning(unittest.TestCase):
    def test(self):
        res = subprocess.Popen("/bin/ps ho rss -C dplane".split(), stdout=subprocess.PIPE)
        out = res.communicate()[0]
        self.assertTrue(out != "", "Run /etc/init.d/dplane start")
        
class dplaneIsReadyToRecord(unittest.TestCase):
    def test(self):
        ret = cplanecmd('status?;')
        self.assertTrue(ret == '!status?0:0:0x3333301;' or ret == '!status?0:0:0x3333311;')

# will only check them as defined by cplane not dplane
class InputStreamsAreCorrect(unittest.TestCase):
    def test(self):
        streams = cplanecmd('input_stream?;')
        (s1, s2) = input_streams.items()
        s1s2 = re.match('!input_stream\?0:0.+vdif:8224:50:42:%s.+%s.+vdif:8224:50:42:%s.+%s;' % (s1[0], s1[1], s2[0], s2[1]), streams)
        s2s1 = re.match('!input_stream\?0:0.+vdif:8224:50:42:%s.+%s.+vdif:8224:50:42:%s.+%s;' % (s2[0], s2[1], s1[0], s1[1]), streams)
        self.assertTrue(s1s2 is not None or s2s1 is not None, "input streams not consistent with input_streams defined at top")

class R2PPSNearSystemClock(unittest.TestCase):
    def test(self):
        if os.getuid() != 0:
            raise OSError("Must be root")
        res3 = subprocess.Popen("/home/oper/bin/vv -i eth3 -p 4001 -n 1".split(), stdout=subprocess.PIPE)
        eth3dt = float(res3.communicate()[0][:-2].split()[-1])
        res5 = subprocess.Popen("/home/oper/bin/vv -i eth5 -p 4001 -n 1".split(), stdout=subprocess.PIPE)
        eth5dt = float(res5.communicate()[0][:-2].split()[-1])
        self.assertTrue(abs(eth3dt) < vv_threshold and abs(eth5dt) < vv_threshold,
            "packet times at (%.3f, %.3f), check GPS lock and NTP" % (eth3dt, eth5dt))

class LastScanCheckOK(unittest.TestCase):
    def test(self):
        sc = cplanecmd('scan_check?;')
        if 'unk' in sc:
            raise ValueError('cplane confused about last scan, perhaps record pending: ' + sc)
        self.assertTrue('OK' in sc)
        
if __name__ == '__main__':
    unittest.main()
