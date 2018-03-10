import json
import pickle
import csv
import numpy as np
import string
#from collections import Counter
#import collections
#import pytorch
import re
def normalizeString(s):
	s = s.lower().strip()
	#s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z]+", r" ", s)
	return s
def to_csv(filename, output):
	cnt = {}
	bi_cnt = {}
	i = 0
	empty_count = 0
	#out = open(output, 'w')
	with open(filename, 'r') as f:#, open(output, 'w') as w:
		for line in f:
			i += 1
			print(i)
			#if i > 1000:
			#	break
			data = json.loads(line)
			if data['title'] == '':
				empty_count += 1
				continue
			title = normalizeString(data['title'])
			if title.find(' at ') != -1:
				title = title[:title.find(' at ')]
			if title.find(' for ') != -1:
				title = title[:title.find(' for ')]
			if title.find(' in ') != -1:
				title = title[:title.find(' in ')]
			if title.find(' of ') != -1:
				title = title[:title.find(' of ')]
			words = title.split()
			for ii in range(len(words)):
				item = words[ii]
				if item in cnt:
					cnt[item] += 1
				else:
					cnt[item] = 1
				if ii == (len(words) - 1):
					continue
				bi_words = words[ii] + ' ' + words[ii+1]
				if bi_words in bi_cnt:
					bi_cnt[bi_words] += 1
				else:
					bi_cnt[bi_words] = 1
			#if title in cnt:
			#	cnt[title] += 1
			#else:
			#	cnt[title] = 1
		#	out.write(line)
		#	if i > 1000:
		#		break
	#out.close()
	print("empty count")
	print(empty_count)
	print("total words")
	print(len(cnt))
	print("total bigram")
	print(len(bi_cnt))
	with open(output+'_unigram'+'.pickle', 'wb') as title_file:
		pickle.dump(cnt, title_file)
	with open(output+'_bigram'+'.pickle', 'wb') as title_file:
                pickle.dump(bi_cnt, title_file)	

def load(filename,outputname):
	with open(filename, 'r') as f:
		dic = pickle.load(f)
		#print(dic)
		print(len(dic))
	topk = 0
	with open(outputname, 'wb') as output:
		for key, value in sorted(dic.items(), key=lambda x: x[1], reverse=True):
			output.write(key + " : " + str(value) + "\n")
			topk += 1
			if topk >10000:
				break

filename = 'json_usa_nyu_09142017.txt'
output = 'data_title_words'
#to_csv(filename, output)
#load(output + '_unigram.pickle', "title_unigram")
load(output + '_bigram.pickle', "title_bigram")
