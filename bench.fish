echo "==========START COMPILE========="
cargo build --release --bin do_query 2>/dev/null 
cp target/release/do_query ~/repo/boolean-query-benchmark/engines/fastful-search/ 
echo "==========START BENCH==========="
pushd .
cd ~/repo/boolean-query-benchmark
for i in (seq 1 19)
	set start_time (date +%s)
	make bench CORPUS_TYPE=$i >> /dev/null
	set end_time (date +%s)
	set execution_time (math "$end_time - $start_time")
	echo "bench CORPUS $i, consume $execution_time seconds"
end
python3 result_plot.py intersection
popd

echo "=======END BENCH!=========="
