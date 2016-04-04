import adc5g, corr
from time import sleep
from datetime import datetime, time, timedelta

roach2 = corr.katcp_wrapper.FpgaClient('r2dbe-1')
roach2.wait_connected()

sblookup = {0:'LSB', 1:'USB'}
pollookup = {0:'X/L', 1:'Y/R'}

sbmap = {'LSB':0, 'USB':1}
polmap = {'X':0, 'Y':1, 'L':0, 'R':1, 'LCP':0, 'RCP':1}

# IF0:station IF0:Sky_SB IF0:BDC_SB IF0:POL IF1:station IF1:Sky_SB IF1:BDC_SB IF1:POL | comment
#0  1   2   3   4  5   6   7
configs = """
Lm LSB LSB LCP Ln LSB LSB RCP | 4-6 interim config
Lm LSB USB LCP Ln LSB USB RCP | 6-8 interim config
Lm USB LSB LCP Ln LSB LSB RCP | 4-6 ALMA config
Lm USB USB LCP Ln LSB USB RCP | 6-8 ALMA config
"""

choices = configs.strip().split('\n')

print "Please choose a configuration [1-%d]:" % (len(choices))
for (i, line) in enumerate(choices):
    print "%3d) %s" % (i+1, line)

i = raw_input("-> ")
choice = choices[int(i)-1]

par = choice.split('|')[0].strip().split()

# set variables
station_id_0 = par[0]
station_id_1 = par[4]
skysb_block_0 = sbmap[par[1]]
skysb_block_1 = sbmap[par[5]]
bdcsb_block_0 = sbmap[par[2]]
bdcsb_block_1 = sbmap[par[6]]
pol_block_0 = polmap[par[3]]
pol_block_1 = polmap[par[7]]

print "setting IF parameters:"
print "  IF0: %s (%s) - Sky:%s, BDC:%s" % (par[0], par[3], par[1], par[2])
print "  IF1: %s (%s) - Sky:%s, BDC:%s" % (par[4], par[7], par[5], par[6])

ref_ep_num = 32 #2014 part 2 = 29
ref_ep_date = datetime(2016,1,1,0,0,0) # date of start of epoch July 1 2014
utcnow = datetime.utcnow()
wait = (1500000 - utcnow.microsecond) % 1e6
sleep(wait / 1e6)
utcnow = datetime.utcnow()

delta       = utcnow-ref_ep_date
sec_ref_ep  = delta.seconds + 24*3600*delta.days
nday = sec_ref_ep/24/3600 # LLB: I believe nday here is off by 1 (0 indexed)
roach2.write_int('r2dbe_vdif_0_hdr_w0_reset',1)
roach2.write_int('r2dbe_vdif_0_hdr_w0_reset',0)
roach2.write_int('r2dbe_vdif_1_hdr_w0_reset',1)
roach2.write_int('r2dbe_vdif_1_hdr_w0_reset',0)
roach2.write_int('r2dbe_vdif_0_hdr_w0_sec_ref_ep',sec_ref_ep)
roach2.write_int('r2dbe_vdif_1_hdr_w0_sec_ref_ep',sec_ref_ep)

roach2.write_int('r2dbe_vdif_0_hdr_w1_ref_ep',ref_ep_num)
roach2.write_int('r2dbe_vdif_1_hdr_w1_ref_ep',ref_ep_num)

############
#   W3
############
# roach2.write_int('r2dbe_vdif_0_hdr_w3_thread_id', thread_id_0)
# roach2.write_int('r2dbe_vdif_1_hdr_w3_thread_id', thread_id_1)

# convert chars to 16 bit int
st0 = ord(station_id_0[0])*2**8 + ord(station_id_0[1])
st1 = ord(station_id_1[0])*2**8 + ord(station_id_1[1])

roach2.write_int('r2dbe_vdif_0_hdr_w3_station_id', st0)
roach2.write_int('r2dbe_vdif_1_hdr_w3_station_id', st1)

############
#   W4
############

eud_vers = 0x02

w4_0 = eud_vers*2**24 + skysb_block_0*4 + bdcsb_block_0*2 + pol_block_0
w4_1 = eud_vers*2**24 + skysb_block_1*4 + bdcsb_block_1*2 + pol_block_1

roach2.write_int('r2dbe_vdif_0_hdr_w4',w4_0)
roach2.write_int('r2dbe_vdif_1_hdr_w4',w4_1)

