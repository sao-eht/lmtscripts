# missing tests:
#   vv time offset (needs root)

import unittest
import corr
import httplib
import os
import numpy as np
import adc5g
import time
import subprocess
from datetime import datetime, timedelta
from pkg_resources import parse_version

def get_switch_ip():
    code = open("/usr/local/src/r2dbe/software/switch_set_IF.py").read()
    return code.split('"')[1]

r2_hostname = 'r2dbe-1'
switch_ip = get_switch_ip()
m6_software_version = '1.2j'
r2_bitcode_md5sum = '6421249e83aa86a9f2630b2c2ea04d22'
if_power_tol = 10 # deviation of std from ideal in ADC 8bit units
r2_threshold_tol = 4 # deviation of th from ideal in ADC 8bit units

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
        self.assertEqual(epoch, roach2.read_int('r2dbe_vdif_0_hdr_w1_ref_ep'))
        self.assertEqual(epoch, roach2.read_int('r2dbe_vdif_1_hdr_w1_ref_ep'))

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
        self.assertEqual(md5, r2_bitcode_md5sum)

class R2IsConnected(unittest.TestCase):
    def test(self):
        self.assertTrue(roach2.is_connected())

class R2GpsIncrementedBy1(unittest.TestCase):
    def test(self):
        utcnow = datetime.utcnow()
        wait = (1500000 - utcnow.microsecond) % 1e6 # get to 0.5s boundary
        time.sleep(wait / 1e6)
        gpspps1 = roach2.read_uint('r2dbe_onepps_gps_pps_cnt')
        time.sleep(1.0)
        gpspps2 = roach2.read_uint('r2dbe_onepps_gps_pps_cnt')
        self.assertTrue(gpspps2 == gpspps1 + 1)

class R2ClockIs256MHz(unittest.TestCase):
    def test(self):
        a=roach2.read_uint('sys_clkcounter')
        b=roach2.read_uint('sys_clkcounter')
        time.sleep(0.1)
        c=roach2.read_uint('sys_clkcounter')
        clk = (c-2*b+a)/1e5
        self.assertTrue(abs(clk - 256.) < 2.0)

class IFPowerIsGood(unittest.TestCase):
    def test(self):
        self.assertTrue(np.abs(np.std(x0) - 30.) <= if_power_tol and
                np.abs(np.std(x1) - 30.) <= if_power_tol)

class IFThresholdIsGood(unittest.TestCase):
    def test(self):
        self.assertTrue(np.abs(np.std(x0) - th0) <= r2_threshold_tol and
                np.abs(np.std(x1) - th1) <= r2_threshold_tol)

class SwitchIsSetToIF(unittest.TestCase):
    def test(self):
        connection = httplib.HTTPConnection(switch_ip,80)
        connection.request("GET","/SWPORT?")
        response = connection.getresponse()
        data = response.read()
        self.assertEqual(int(data), 0)

class Mark6SoftwareIsCurrent(unittest.TestCase):
    def test(self):
        self.assertTrue(os.path.exists('/usr/local/src/Mark6_%s' % m6_software_version))

class TimezoneIsUTC(unittest.TestCase):
    def test(self):
        self.assertEqual(time.tzname[1], 'UTC')

class cplaneIsRunningAndUnder1GB(unittest.TestCase):
    def test(self):
        res = subprocess.Popen("/bin/ps ho rss -C cplane".split(), stdout=subprocess.PIPE)
        out = res.communicate()[0]
        self.assertTrue(out != "" and int(out) < 1e6)

class dplaneIsRunning(unittest.TestCase):
    def test(self):
        res = subprocess.Popen("/bin/ps ho rss -C dplane".split(), stdout=subprocess.PIPE)
        out = res.communicate()[0]
        self.assertTrue(out != "")
        
if __name__ == '__main__':
    unittest.main()
