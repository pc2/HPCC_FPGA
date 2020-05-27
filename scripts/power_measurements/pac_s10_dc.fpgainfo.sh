#!/bin/bash

LOGFILE=powermeasure.csv

echo "" > $LOGFILE


# Start the benchmark

$@ &

bm_pid=$!
# Start power measurements

while $(kill -0 $bm_pid); do
    echo $(fpgainfo power | grep "Amps\|Volts" | sed -r 's/.*: ([0-9]+)\.([0-9]+).*/\1.\2/g' | sed -r ':a;N;$!ba;s/\n/,/g') >> $LOGFILE
    sleep 0.01
done