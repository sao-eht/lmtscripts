#!/usr/bin/env python

# monitor for EHT observations
# based on Helge's start_eht script
# 2018.04.22 LLB

import sys
from subprocess import check_output
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep

xml_schedule = sys.argv[1]

prescan = 5.   # seconds before scan to do stuff

r2dbes = ['r2dbe1', 'r2dbe2', 'r2dbe3', 'r2dbe4']
alc = "/home/oper/bin/alc.py"

tree = ET.parse(xml_schedule)
root = tree.getroot()
wipe = 30 * '\b' + 30 * ' ' + 30 * '\b'

predt = timedelta(seconds=prescan)

def timersleep(until, line='waiting.. '):
    now = datetime.utcnow()
    if now > until:
        return False
    sys.stdout.write(wipe)
    while now < until:
        wait = (until - now).total_seconds()
        sys.stdout.write('\r%s%.0f ' % (line, wait))
        sys.stdout.flush()
        sleep(0.1 + wait % 1.0)
        now = datetime.utcnow()
    return True

def runalc(r2dbes, line='alc ..'):
    sys.stdout.write('%s\r%s' % (wipe, line))
    for r2 in r2dbes:
        out = check_output([alc, r2])
        (th0, th1) = (int(a[4:].replace(',','')) for a in out.split('\n')[2].split()[-2:])
        sys.stdout.write('%d:%d ' % (th0, th1))
        sys.stdout.flush()

for scan in root.findall("scan"):

    name     = scan.get('scan_name')
    duration = int(scan.get('duration'))
    source   = scan.get('source')
    start    = datetime.strptime(scan.get('start_time'), '%Y%j%H%M%S')
    stop     = start + timedelta(seconds=duration)

    if timersleep(start - predt, line='[%s %s] alc in.. ' % (name, source)):
        runalc(r2dbes, line='[%s %s] alc .. ' % (name, source))
    timersleep(start, line='[%s %s] starts in.. ' % (name, source))
    timersleep(stop, line='[%s %s] recording.. ' % (name, source))

