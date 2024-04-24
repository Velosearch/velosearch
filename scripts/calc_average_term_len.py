import json
import os
import tqdm
import sys
from nltk import word_tokenize
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

def get_words(_sent):

    return [word for word in word_tokenize(_sent) if word not in stopWords] 

corpus_files = []
corpus_path = "/home/RT_Enzyme/repo/docker/spark/share/wikipedia/corpus"
prefix = sys.argv[1]
print("prefix:", prefix)
for f in os.listdir(corpus_path):
    path = os.path.join(corpus_path, f)
    if os.path.isdir(path):
        if f.startswith(prefix):
            for s in os.listdir(path):
                corpus_files.append(os.path.join(path,s))
    else:
        corpus_files.append(os.path.join(path))

words = []
words_len_dict = {}

def preprocess_wiki(jsons):
    """
    预处理大语料wiki
    :param jsons: 输入的List[json]，只获取每个json中的text属性
    :return: 返回分词后的语料列表
    """
    for item in jsons:
        if 'text' in item:
            text = item['text']
            tokenized_words = get_words(text)
            for word in tokenized_words:
                word_len = len(word.encode())
                if word_len in words_len_dict:
                    words_len_dict[word_len] += 1
                else:
                    words_len_dict[word_len] = 1
        else:
            continue

def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return list(map(lambda l: json.loads(l), lines))
    
for f in tqdm.tqdm(corpus_files):
    preprocess_wiki(read_file(f))

json.dump(words_len_dict, open(f"term_len_{prefix}.json", "w"))

