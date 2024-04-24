#!/bin/bash
PID=`ps -ef | grep concurrency_bench | grep -v 'grep' | awk '{print $2}'`
if [ ${#PID} -eq 0 ]
then
    echo "databend-query not running"
    exit -1
fi

echo ${PID}

flamegraph --root --pid ${PID}