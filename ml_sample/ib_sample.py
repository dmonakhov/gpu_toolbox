#!/usr/bin/python3

import argparse
import datetime
import os
import time
import sys

HW_COUNTERS = ['recv_bytes', 'recv_wrs', 'send_bytes', 'send_wrs']
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/lifespan:12
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rdma_read_bytes:83829566349996
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rdma_read_resp_bytes:82752269975546
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rdma_read_wr_err:0
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rdma_read_wrs:888872032
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/recv_bytes:1636854923852
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/recv_wrs:887511660
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rx_bytes:85466418782288
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rx_drops:0
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/rx_pkts:1776383627
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/send_bytes:1665042978528
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/send_wrs:890700972
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/tx_bytes:84417311044934
#/sys/class/infiniband/rdmap144s27/ports/1/hw_counters/tx_pkts:11002381069
# timeout 2 nvidia-smi -lms 10 -i 0 --query-gpu=timestamp,pci.bus_id,power.draw,utilization.gpu --format=csv
# 2023/07/16 19:57:29.292, 00000000:10:1C.0, 125.72 W, 99 %

def read_one(fds):
    ret = []
    for fd in fds:
        ret.append(os.pread(fd, 20, 0).decode().split()[0])
    return ret

def read_loop(fds, delay, num):
    init = 1
    data_sz = len(fds)
    prev_ts = datetime.datetime.now().timestamp()
    prev_data = read_one(fds)

    for i in range(num):
        time.sleep(delay)
        cur_ts = datetime.datetime.now().timestamp()
        cur_data = read_one(fds)
        ret = []
        for f in range(data_sz):
            ret.append(cur_data[f])
            ret.append(str(int(cur_data[f]) - int(prev_data[f])))
        print(cur_ts, round((cur_ts - prev_ts)*1000), '\t'.join(ret))
        prev_ts = cur_ts
        prev_data = cur_data


def main(argv):
    parser = argparse.ArgumentParser(
                    prog='ib_sample',
                    description='sample infiniband data')
    parser.add_argument('interface', help='interface to sample')
    parser.add_argument('-lms', default=100, type=int, help='Report query data at the specified interval in miliseconds')
    parser.add_argument('-n', '--samples', type=int, default = 100, help='number of samples')
    parser.add_argument('-p', '--port', default=1, type=int, help='interface port')
    parser.add_argument('-c', '--counters', default = HW_COUNTERS, action='append')
    args = parser.parse_args(argv)
    fds = []
    names = [ ]
    for ct in args.counters:
        fname = '/sys/class/infiniband/' + args.interface + '/ports/' + str(args.port) + '/hw_counters/' + ct
        fd = os.open(fname, os.O_RDONLY)
        fds.append(fd)
        names.append(ct)
        names.append("dif:" + ct)
    print("#", "timestamp", "delta_ms", "\t".join(names))

    read_loop(fds, args.lms / 1000.0, args.samples)


if __name__ == "__main__":
    main(sys.argv[1:])
