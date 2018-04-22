#!/usr/bin/env python

# usage: python xml2human_v2.py <filename.xml>
# will parse the xml file for variables below, 
# and print them to a csv file named filename.csv

import subprocess 
import sys
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep
from optparse import OptionParser

import csv


parser = OptionParser()
(options, args) = parser.parse_args()


# get schedule file
schedule = args[0]

tree = ET.parse(schedule)
root = tree.getroot()

name = schedule.split('.')

f=open(name[0]+'.csv','wb')

mywriter = csv.writer(f)
mywriter.writerow(('Scan #','Scan name', 'Source','Start UTC','Stop UTC','Duration (min)','Downtime (min)','Tsys (LCP; RCP)', 'Tau', 'Pointing Offsets (Az, El)', 'Phasing Efficiency', 'Local Obs. Number', 'Comments'))

k = 1
for scan in root.findall("scan"):

	duration = int(scan.get('duration'))
        src      = scan.get('source')
	dtStart  = datetime.strptime(scan.get('start_time'), '%Y%j%H%M%S')
	dtStop   = dtStart + timedelta(seconds=duration)
	durmin   = float(duration)/60
	scanName = scan.get('scan_name')

        if k>1:
            dt       = dtStart-dtStop_last
            offtime  = float(dt.seconds)/60

            mywriter.writerow((k-1,scanName_last, src_last,dtStart_last,dtStop_last,durmin_last,offtime))

        k = k+1 
        dur_last = duration
        src_last = src
        dtStart_last = dtStart
        dtStop_last = dtStop
        durmin_last=durmin
        scanName_last = scanName

mywriter.writerow((k-1,scanName_last, src_last,dtStart_last,dtStop_last,durmin_last,offtime))
f.close()
