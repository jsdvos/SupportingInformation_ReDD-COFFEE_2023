#! /usr/bin/python
import re, os

from optparse import OptionParser


parser = OptionParser("""Create a pbs .sh-file based on a gaussian input file
Usage: js-g16.py [options] comfile1 [comfile2 ...]

Example usages:
js-g16.py job.com
js-g16.py job1.com job2.com
js-g16.py */*.com
find . | grep '\.com$' | xargs js-g09.py

Use -h or --help for options.""")
parser.add_option(
    "--time", "-t", default=None,
    help="The wall clock limit in hours or in hours:minutes. The default is the maximum for the given que.",
)
parser.add_option(
    "--qname", "-q", default="long", help="the name of the que"
)
parser.add_option(
    "--mail", "-m", default="n", help="mail option (b (begins exec), e (termination exec), a (aborted exec))"
)

(options, args) = parser.parse_args()
if len(args) > 0:
    com_filenames = args
else:
    parser.error("Expecting at least one argument.")

if options.time is not None:
    try:
        if ':' in options.time:
            words = options.time.split(':')
            if len(words) != 2:
                raise ValueError
            hours = int(words[0])
            minutes = int(words[1])
        else:
            hours = int(options.time)
            minutes = 0
    except ValueError:
        parser.error("the time option must be an integer or two integers separated by a colon.")
else:
    if options.qname == "long":
        hours = 72
        minutes = 0
    elif options.qname == "short":
        hours = 11
        minutes = 0
    elif options.qname == "debug":
        hours = 0
        minutes = 59
    else:
        parser.error("Maximum wall time not known for queue %s" % option.qname)

re_mem = re.compile("%mem\s*=\s*(?P<mem>\d+\S*)\s*$", re.IGNORECASE)
re_nproc = re.compile("%nproc\s*=\s*(?P<nproc>\d+)\s*$", re.IGNORECASE)

sh_template = """#!/bin/sh
#
#PBS -N _%(basename)s
#PBS -l walltime=%(hours)i:%(minutes)s:00
#PBS -l nodes=1:ppn=%(nproc)i
#PBS -l vmem=%(vmem)s
#PBS -m %(mail)s
#

# Prepare for Gaussian computation
newgrp - ggaussian
export MKL_NUM_THREADS=1
export KMP_STACKSIZE=16m
export G09NOHOARD=1
export MODULEPATH=/apps/gent/SL6/sandybridge/modules/all:$MODULEPATH
module load Gaussian/g16_C.01-intel-2019b

# Set up Gaussian input
ORIGDIR=$PBS_O_WORKDIR
WORKDIR=/local/$PBS_JOBID
mkdir -p $WORKDIR
cd $WORKDIR
if [ -e $ORIGDIR/%(basename)s.in.fchk ]; then
  cp $ORIGDIR/%(basename)s.in.fchk $WORKDIR
  unfchk %(basename)s.in.fchk %(basename)s.chk
fi
cp $ORIGDIR/%(basename)s.com $WORKDIR

# Copy back results every 30 minutes
( while true; do
    sleep 1800
    cp %(basename)s.log $ORIGDIR
    cp %(basename)s.chk $ORIGDIR
  done ) &

# Run Gaussian and save PID
(time g16 < %(basename)s.com > %(basename)s.log; rm gaussian.pid) &
PID=$!
echo $PID > gaussian.pid

# Wait till wall time is nearing. If not done: kill and start copying back.
( sleep %(grace)i ; if [[ -e gaussian.pid ]]; then kill -9 $pid; fi; ) &
KILLERPID=$!
wait $PID
kill $KILLERPID

# Convert checkpoint file and copy back
formchk %(basename)s.chk %(basename)s.fchk && rm %(basename)s.chk
cp %(basename)s.* $ORIGDIR

rm -rf $WORKDIR

"""


def read_size(s):
    if s.upper().endswith("GB"):
        return int(s[:-2])*1024**3
    elif s.upper().endswith("MB"):
        return int(s[:-2])*1024**2

def write_size(s):
    if s < 20*1024**2:
        return str(s)
    elif s < 20*1024**3:
        return str("%iMB" % (s/1024/1024))
    else:
        return str("%iGB" % (s/1024/1024/1024))


for com_filename in com_filenames:
    prefix = com_filename
    if prefix.endswith(".com"):
        prefix = prefix[:-4]
    
    basename = os.path.basename(prefix)
    #dirname = os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(prefix)))

    f = open(com_filename, 'r')
    nproc = None
    memory = None
    found_maxdisk = False
    for line in f:
        match = re_mem.search(line)
        if match is not None:
            memory = read_size(match.group("mem"))
        match = re_nproc.search(line)
        if match is not None:
            nproc = int(match.group("nproc"))
        if not found_maxdisk and 'maxdisk' in line:
            found_maxdisk = True
    f.close()
    if not found_maxdisk:
        print("WARNING no maxdisk specified!")
    if nproc is None or memory is None:
        parser.error("Could not find %%nproc or %%mem line in file %s. Is this a gaussian com file?" % com_filename)

    #for var in "VSC_HOME", "VSC_DATA", "VSC_SCRATCH":
    #    dirname = dirname.replace(os.getenv(var), "${%s}" % var)

    sh_contents = sh_template % {
        "basename": basename, 
        "nproc": nproc, 
        "hours": hours, 
        "minutes": minutes, 
        #"origdir": dirname,
        "qname": options.qname,
        "grace": hours*3600 + minutes*60 - 15*60, # stop 15 minutes before end of wall time
        "vmem": write_size(memory*1.3 + 1024**3),
        "mail": options.mail
    }


    f = open("%s.sh" % prefix, "w")
    f.write(sh_contents)
    f.close()
    print("Written %s.sh" % prefix)

