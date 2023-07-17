#!/bin/bash

set -ue

TIME=${1:-10}
NET=${2:-rdmap16s27}
GPU=${3:-0}
SAMPLE_MS=30

if [ "$TIME" == "-h" ] || [ "$TIME" == "--help" ]
then
    echo "Usage: $0 [SAMPLE_TIME_SEC] [RDMA_DEV] [GPU_DEV]"
    exit 1
fi

TESTID=logs/ml_sample-$(date +%Y%m%d-%H%M%S)
mkdir $TESTID
echo "ML sample: time:$TIME net:$NET gpu:$GPU log:`realpath $TESTID`"

nvidia-smi > $TESTID/nvidia_smi.log
hostname > $TESTID/hostname.log
cp ib_sample.py  csv_ts2epoh.py ml_sample.plot $TESTID/
cp $0 $TESTID/ml_sample.sh

timeout $((TIME+1)) nvidia-smi -lms $SAMPLE_MS -i $GPU --query-gpu=timestamp,pci.bus_id,power.draw,utilization.gpu --format=csv > $TESTID/gpu_raw.txt &
sleep 1
./ib_sample.py -lms $SAMPLE_MS  -n $(((TIME*1000) / $SAMPLE_MS)) $NET > $TESTID/net.txt
wait

cat $TESTID/gpu_raw.txt |  $TESTID/csv_ts2epoh.py > $TESTID/gpu.txt

