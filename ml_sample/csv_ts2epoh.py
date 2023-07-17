#!/usr/bin/env python3

from datetime import datetime 
import sys
import argparse

# Usage
#nvidia-smi -lms 30 -i 0 --query-gpu=timestamp,pci.bus_id,power.draw,utilization.gpu --format=csv | csv_ts2epoh.py

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--field', type=int, default=1, help='timstamp field')
parser.add_argument('-d', '--delimiter', default=',', help='field delimiter')
parser.add_argument('-t','--time-format', default='%Y/%m/%d %H:%M:%S.%f', help='datetime format')
args = parser.parse_args()
fnum=args.field -1

for line in sys.stdin:
    tk = line.strip().split(args.delimiter)
    try:
        d = datetime.strptime(tk[fnum], args.time_format)
        tk[fnum] = "{:.6f}".format(d.timestamp())
    except Exception as e:
        pass
    print(args.delimiter.join(tk))
