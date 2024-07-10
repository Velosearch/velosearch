#!/bin/bash
PID=`ps -ef | grep do_query | grep -v 'grep' | awk '{print $2}'`
if [ ${#PID} -eq 0 ]
then
    echo "databend-query not running"
    exit -1
fi

echo ${PID}

flamegraph --root --pid ${PID}