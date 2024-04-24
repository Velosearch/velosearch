import os
import subprocess
import tqdm

file_path = "/home/RT_Enzyme/repo/docker/spark/share/wikipedia/corpus/"
file_list = list(os.listdir(file_path))
file_list.sort()

times = []

for num in tqdm.tqdm(range(1, 21)):
    append_file = file_list[0: 5]
    res = subprocess.run(
        ["/home/RT_Enzyme/repo/fastfull-search/target/release/fastfull-search", "-h", "posting-table",  "--partition-num", "1", "-b", f"{num * 512}",  "--base", f"{file_path}",  *append_file],
        stdout=subprocess.PIPE)
    res = int(res.stdout)
    print(res)
    times.append(res)

print(times)