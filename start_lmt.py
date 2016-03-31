#!/usr/bin/env python

###########################################################################
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###########################################################################
# Original author: Helge Rottmann (MPIfR) rottmann(AT)mpifr.de
# Edited for LMT by Laura Vertatschitsch and Lindy Blackburn
###########################################################################

import subprocess 
import sys
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep
from optparse import OptionParser



alcCommand = "/usr/local/src/r2dbe/software/alc.py"
switchCommand = "/usr/local/src/r2dbe/software/switch_set_IF.py"
monCommand = "/usr/local/src/r2dbe/software/r2dbe_monitor.py"
m6ccCommand = "/usr/local/bin/M6_CC"

version = "1.0"


def usage():
	usage = "USAGE: "
	usage += "%prog [OPTIONS] xml-schedule \n\n"
	usage += "This script launches the EHT observation schedule\n"
	usage += "In the scan gaps the following additional tasks are being performed:\n"
	usage += "1) R2DBE 2-bit requantization is done\n"
	usage += "2) noise generator output is set to IF\n"
	usage += "3) r2dbe_monitor is displayed (automatically ends before the start of the next scan)\n\n"
	

	return(usage)

def validate():
	# check that schedule file exists
	if not os.path.isfile(args[0]):
		sys.exit ("schedule file does not exist: %s" % args[0])

	# check that scripts exist:
	for script in [alcCommand, switchCommand, monCommand, m6ccCommand]:
		if not os.path.isfile(script):
			sys.exit ("Required file does not exist: %s" % script)


	


def executeJob(command, arguments, message):

	print message
	return_code = subprocess.call([command, arguments], shell=False)
        if return_code != 0:
                print "Error executing: %s" % command

	return

def isValidTime(scanStart, scanStop):

	dtNow = datetime.utcnow()

	# Check if we are too close to the beginning of the next scan
	# or within the next scan

	if dtNow + timedelta(seconds=preScanMargin) < scanStart:
		return(True)

	#if dtNow < scanStop + timedelta(seconds=postScanMargin):
	#	print " within the next scan"
	#	return(False)
#
	return(False)

# get floating point seconds
def precise_seconds(dt):
    sec = dt.seconds + 1e-6*dt.microseconds
    return sec


# MAIN
parser = OptionParser(usage=usage(),version=version)

parser.add_option("--preScanMargin", type="int", default=15, dest="preScanMargin", help="number of seconds before the beginning of the next scan in which no further commands will be executed (default=15)")
parser.add_option("--postScanMargin", type="int", default=05, dest="postScanMargin", help="number of seconds after the end of the next scan in which no further commands will be executed (default=05)")

(options, args) = parser.parse_args()

if len(args) != 1:
	parser.print_help()
	sys.exit(1)

schedule = args[0]
postScanMargin = options.postScanMargin
preScanMargin = options.preScanMargin

validate()

# launch the mark6 schedule
#schedArgs = "%s -f %s" % (m6ccCommand, schedule)
#m6cc = subprocess.Popen(["xterm", "-e", schedArgs], stderr=subprocess.STDOUT)

tree = ET.parse(args[0])
root = tree.getroot()

for scan in root.findall("scan"):

	duration = int(scan.get('duration'))

	dtStart = datetime.strptime(scan.get('start_time'), '%Y%j%H%M%S')
	dtStop = dtStart + timedelta(seconds=duration)
	dtNow = datetime.utcnow()
	
	if dtStop < dtNow:
		print "Scan %s lies in the past. Skipping" % scan.get('scan_name')
		continue

	if isValidTime(dtStart, dtStop):
		print "Next scan (%s) will run from %s to %s" % (scan.get('scan_name'), dtStart, dtStop)
		
		executeJob("python",switchCommand, "making sure the noise source output is switched to IF")
		executeJob("python", alcCommand, "starting R2DBE requantization")
		p = subprocess.Popen(["python", monCommand], stdout=subprocess.PIPE)

		# time(sec) until beginning of the next scan
		deltaStart = dtStart  - timedelta(seconds=preScanMargin) - datetime.utcnow()
		
		# sleep until the beginning of the scan (minus safety margin)
		print "sleeping until: ", datetime.utcnow() + deltaStart	
        #print "%f vs %f" %(float(deltaStart.seconds),precise_seconds(deltaStart))
		#sleep(float(deltaStart.seconds))
		sleep(precise_seconds(deltaStart))

		# kill the r2db_monitor (should not be running during recording)
		p.terminate()
		executeJob("python", switchCommand, "making sure the noise source output is switched to IF")

        # wait until just 2 second before a scan
		deltaStart = dtStart  - timedelta(seconds=2) - datetime.utcnow()
		print "sleeping until: ", datetime.utcnow() + deltaStart	

        # run alc.py
		#sleep(float(deltaStart.seconds))
		sleep(precise_seconds(deltaStart))
		executeJob("python", alcCommand, "starting R2DBE requantization")

		# time(sec) until end of the next scan
		deltaStop = dtStop  + timedelta(seconds=postScanMargin) - datetime.utcnow()

		print "sleeping until: ", datetime.utcnow() + deltaStop
		#sleep(float(deltaStop.seconds))
		sleep(precise_seconds(deltaStop))
		


