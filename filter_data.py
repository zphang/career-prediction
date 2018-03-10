'''
filter data with title not in labels
'''
import re
import pickle
import argparse
import logging
from os.path import join as pjoin
from datetime import datetime
import json
import heapq
def get_labels(filename, topk, stopwords):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
        for k, v in list(dic.items()):
            if len(k) <= 1 or k in stopwords:
                del dic[k]
    print("total grams:", len(dic), ". getting top ", topk)
    keys = heapq.nlargest(topk, dic, key=dic.get)
    return keys

def get_time_str():
	time_str = str(datetime.now())
	time_str = '_'.join(time_str.split())
	time_str = '_'.join(time_str.split('.'))
	time_str = ''.join(time_str.split(':'))
	return time_str

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s

def in_classes(line, labels):
	data = json.loads(line)
	if data["title"] == '':
		return False
	title_str = normalize_string(data["title"])
	for w in title_str.split():
		if w in labels:
			return True
	return False

parser = argparse.ArgumentParser()
parser.add_argument('-corpus', '--corpus', type=str, default='')
parser.add_argument('-output', '--output', type=str, default='')
parser.add_argument("-grams_file", "--grams_file", default='data_title_words_unigram.pickle',
	help="location of the unigram/bigram(extracted from title) file")
parser.add_argument("-topk_title", "--topk_title", default=5000, type=int, 
	help="topk words appeared in title, used as the #classifications for title")
parser.add_argument("-stopwords_file", "--stopwords_file", default='stopwords.tsv',
	help="stopwords file location")
args = parser.parse_args()
log_file = pjoin('process_' + get_time_str() + '.log')
logging.basicConfig(filename=log_file,level=logging.DEBUG)
chunksize = 1000

swfile = args.stopwords_file
with open(swfile) as f:
	stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]
labels = get_labels(args.grams_file, args.topk_title, stopwords)
bunch = []
i = 0
with open(args.corpus, "r") as r, open(args.output, "w") as w:
	for line in r:
		if not in_classes(line, labels):
			continue
		bunch.append(line)
		if len(bunch) == chunksize:
			i += 1
			w.writelines(bunch)
			bunch = []
			logging.info("%s \t  %d", get_time_str(), i*chunksize)
	w.writelines(bunch)
