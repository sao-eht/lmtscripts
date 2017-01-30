LMT Scripts
===========

useful scripts for EHT observations at LMT

#### collect_mark6_data.py


#### corestats.py

print ADC core stats, note default integration time is 10s which gives a very rough measurement (mean to ~0.5)

```
oper@Mark6-4047:~/lindy$ python corestats.py
IF0 Core 0: mean  0.16 std 40.80
IF0 Core 1: mean  0.54 std 40.93
IF0 Core 2: mean  0.68 std 41.00
IF0 Core 3: mean  0.25 std 40.98
IF1 Core 0: mean  0.38 std 41.83
IF1 Core 1: mean  0.37 std 41.99
IF1 Core 2: mean  0.23 std 41.97
IF1 Core 3: mean  0.44 std 41.85
```

#### diskinfo.py

print module disk information (SSCI, SAS, device, partition label)

```
oper@Mark6-4047:~/lindy$ su
Password:
root@Mark6-4047:/home/oper/lindy# python diskinfo.py
[0:0:0:0]   sas:0x4433221100000000 /dev/sdb  EHT%0033_0
[0:0:1:0]   sas:0x4433221101000000 /dev/sdc  EHT%0033_1
[0:0:2:0]   sas:0x4433221102000000 /dev/sdd  EHT%0033_2
[0:0:3:0]   sas:0x4433221103000000 /dev/sde  EHT%0033_3
[0:0:4:0]   sas:0x4433221105000000 /dev/sdf  EHT%0033_5
[0:0:5:0]   sas:0x4433221104000000 /dev/sdg  EHT%0033_4
[0:0:6:0]   sas:0x4433221106000000 /dev/sdh  EHT%0033_6
[0:0:7:0]   sas:0x4433221107000000 /dev/sdi  EHT%0033_7
[0:0:8:0]   sas:0x4433221108000000 /dev/sdj  EHT%0034_0
[0:0:9:0]   sas:0x4433221109000000 /dev/sdk  EHT%0034_1
[0:0:10:0]  sas:0x443322110a000000 /dev/sdl  EHT%0034_2
[0:0:11:0]  sas:0x443322110b000000 /dev/sdm  EHT%0034_3
[0:0:12:0]  sas:0x443322110d000000 /dev/sdn  EHT%0034_5
[0:0:13:0]  sas:0x443322110c000000 /dev/sdo  EHT%0034_4
[0:0:14:0]  sas:0x443322110e000000 /dev/sdp  EHT%0034_6
[0:0:15:0]  sas:0x443322110f000000 /dev/sdq  EHT%0034_7
[17:0:0:0]  sas:0x4433221100000000 /dev/sdr  EHT%0035_0
[17:0:1:0]  sas:0x4433221101000000 /dev/sds  EHT%0035_1
[17:0:2:0]  sas:0x4433221102000000 /dev/sdt  EHT%0035_2
[17:0:3:0]  sas:0x4433221103000000 /dev/sdu  EHT%0035_3
[17:0:4:0]  sas:0x4433221104000000 /dev/sdv  EHT%0035_4
[17:0:5:0]  sas:0x4433221105000000 /dev/sdw  EHT%0035_5
[17:0:6:0]  sas:0x4433221106000000 /dev/sdx  EHT%0035_6
[17:0:7:0]  sas:0x4433221107000000 /dev/sdy  EHT%0035_7
[17:0:8:0]  sas:0x4433221108000000 /dev/sdz  EHT%0036_0
[17:0:9:0]  sas:0x4433221109000000 /dev/sdaa EHT%0036_1
[17:0:10:0] sas:0x443322110a000000 /dev/sdab EHT%0036_2
[17:0:11:0] sas:0x443322110b000000 /dev/sdac EHT%0036_3
[17:0:12:0] sas:0x443322110c000000 /dev/sdad EHT%0036_4
[17:0:13:0] sas:0x443322110d000000 /dev/sdae EHT%0036_5
[17:0:14:0] sas:0x443322110e000000 /dev/sdaf EHT%0036_6
[17:0:15:0] sas:0x443322110f000000 /dev/sdag EHT%0036_7
```

#### mall.py

manually mount disks on Mark6 unit (bypass cplane)

#### mirror.py

mirror.py archives vdif files from source modules onto a backup module on the
same Mark6. Files are copied disk-to-disk to a subdirectory based on the source
minodule MSN.

```
# copy data from 71, 72, 73 into backup directories on 143
python mirror.py BHC%0071 BHC%0072 BHC%0073 BHC%0143 > go.sh
python diskinfo.py # sanity check disk mounting and copy script
. go.sh # run copy script
```

#### og.py

use noise to train and set ADC core offset and gain

```
oper@Mark6-4047:~/lindy$ python /usr/local/src/r2dbe/software/switch_set_noise.py
Command executed successfully.
oper@Mark6-4047:~/lindy$ python og.py

usage:
  python og.py clear (clear previous og registers, otherwise all future solutions will be iterative)
  python og.py 3600 (accumulate 1 hour of counts, calculate solution, and apply if setog is True)
  python og.py ogsol-20150320-134300-3600.npy (apply a saved solution)
```

#### r2dbe_start.py


#### ready.py

unit tests to check readiness of mark6/r2dbe configuration:

```
oper@Mark6-4047:~/lindy$ python ready.py -v
test (__main__.IFPowerIsGood) ... ok
test (__main__.IFThresholdIsGood) ... ok
test (__main__.Mark6SoftwareIsCurrent) ... ok
test (__main__.R2BitcodeIsUpToDate) ... ok
test (__main__.R2IsConnected) ... ok
test (__main__.R2PPSIsNonzero) ... ok
test (__main__.SwitchIsSetToIF) ... ok
test (__main__.TimezoneIsUTC) ... ok
test (__main__.cplaneIsRunningAndUnder1GB) ... ok
test (__main__.dplaneIsRunning) ... ok
----------------------------------------------------------------------
Ran 10 tests in 0.092s
OK
```

#### start_lmt.py

based on Helge's start_eht.py (Starting the schedule) schedule helper script.
edited to provide alc.py just before each scan
