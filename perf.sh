#!/bin/bash
PID=`ps -ef | grep do_query | grep -v 'grep' | awk '{print $2}'`
if [ ${#PID} -eq 0 ]
then
    echo "databend-query not running"
    exit -1
fi

echo ${PID}

perf stat -e branch-misses,cycles,instructions,cache-misses,cache-references,l2_rqsts.demand_data_rd_miss,l2_rqsts.demand_data_rd_hit,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses -p ${PID}
