import  os
import tqdm

file_path = "/home/RT_Enzyme/repo/docker/spark/share/wikipedia/corpus/"
file_list = list(os.listdir(file_path))
file_list.sort()

corpus_len = []

for f in tqdm.tqdm(file_list[:10]):
    p = os.path.join(file_path, f)
    l = 0
    for ff in os.listdir(p):
        with open(os.path.join(p, ff)) as f:
            l += len(f.readlines())
    if len(corpus_len) == 0:
        corpus_len.append(l)
    else:
        corpus_len.append(corpus_len[-1] + l)

print(corpus_len)