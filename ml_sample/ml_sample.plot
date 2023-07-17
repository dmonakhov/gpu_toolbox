#!/usr/bin/gnuplot

set xlabel "Timstamp in seconds"
set ylabel "Power vs Net perf"
#set terminal postscript portrait enhanced mono dashed lw 1 'Helvetica' 14

set grid ytics lt 0 lw 1 lc rgb "#bbbbbb"
set grid xtics lt 0 lw 1 lc rgb "#bbbbbb"
set style line 1 lt 1 lw 3 pt 3 linecolor rgb "red"
set autoscale

# Fine tune data ranges
plot 'net.txt' u 2:1
start_ts = GPVAL_DATA_Y_MIN
stop_ts = GPVAL_DATA_Y_MAX
duration=stop_ts - start_ts
set xrange [0:duration]
print start_ts, stop_ts

plot 'net.txt' using ($1-start_ts):(($4/$2)/1000) with lines title "Net rx GB/sec", \
     'net.txt' using ($1-start_ts):(($8/$2)/1000) with lines title "Net tx GB/sec", \
      'gpu.txt' using ($1-start_ts):3 with lines title "GPU power W"