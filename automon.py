#!/usr/bin/env python

# monitor for EHT observations
# run from control computer
# inspired by Helge's start_eht script
# 2018.04.22 LLB

import sys
import argparse
from subprocess import check_output
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep
import glob

schedules = sorted(glob.glob('e?????.xml'), key=lambda x: x[-6:])
if len(schedules) > 0:
    lastxml = schedules[-1]
    nargs = '?'
else:
    lastxml = None
    nargs = 1

parser = argparse.ArgumentParser()
parser.add_argument('schedule', nargs=nargs, help='xml schedule (from vex2xml)', default=lastxml)
parser.add_argument('-a', '--alc', help='do alc <N> seconds before scan start time (default 5)',
                    nargs='?', const=5., type=float, default=None)
parser.add_argument('-n', '--newlines', help='put each status report on new line', action='store_true', default=False)
args = parser.parse_args()

r2dbes = ['r2dbe1', 'r2dbe2', 'r2dbe3', 'r2dbe4']
alc = "/home/oper/bin/alc.py"

tree = ET.parse(args.schedule)
root = tree.getroot()
if args.newlines:
    preline = '\n'
else:
    preline = 30 * '\b' + 30 * ' ' + 30 * '\b'

def timersleep(until, line='waiting.. '):
    now = datetime.utcnow()
    if now > until:
        return False
    sys.stdout.write(preline)
    while now < until:
        wait = (until - now).total_seconds()
        sys.stdout.write('\r%s%.0f ' % (line, wait))
        sys.stdout.flush()
        sleep(0.1 + wait % 1.0)
        now = datetime.utcnow()
    return True

def runalc(r2dbes, line='alc '):
    sys.stdout.write('%s\r%s' % (preline, line))
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

    if (args.alc is not None) and timersleep(start - timedelta(seconds=float(args.alc)), line='[%s %s] alc in.. ' % (name, source)):
        runalc(r2dbes, line='[%s %s] alc ' % (name, source))
    timersleep(start, line='[%s %s] starts in.. ' % (name, source))
    timersleep(stop, line='[%s %s] recording.. ' % (name, source))

