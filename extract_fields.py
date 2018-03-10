'''
extracts fields from dataset for fasttext classification
'''

import argparse
import pickle
import json
import logging
from os.path import join as pjoin
from utils import *

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
log_file = pjoin('extract_' + get_time_str() + '.log')
logging.basicConfig(filename=log_file,level=logging.DEBUG)
cnt = 0
bunchsize = 10000
swfile = args.stopwords_file
with open(swfile) as f:
	stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]
labels = get_labels(args.grams_file, args.topk_title, stopwords)
bunch = []

with open(args.corpus, "r") as r, open(args.output, "w") as w:
	for line in r:
		data = json.loads(line)
		if data["title"] == "":
			continue
		label = get_classes(data["title"], labels)
		text = ''

		if data["expertise"] != "":
			text += normalize_string(data["expertise"])
			text += ' '

		if data["educations"] != None:
			for edu in data["educations"]:
				if edu["campus"] != None:
					text += (normalize_string(edu["campus"]) + ' ')
				if edu["major"] != None:
					text += (normalize_string(edu["major"]) + ' ')
				if edu["specialization"] != None:
					text += (normalize_string(edu["specialization"]) + ' ')

		if data["awards"] != None:
			for a in data["awards"]:
				text += (normalize_string(a)) + ' '

		if data["certifications"] != None:
			for cert in data["certifications"]:
				if cert["title"] != None:
					text += normalize_string(cert["title"] + ' ')
				if cert["description"] != None:
					text += normalize_string(cert["description"] + ' ')

		if data["languages"] != None:
			for l in data["languages"]:
				text += (l.lower() + ' ')

		if data["experiences"] != None:
			for exp in data["experiences"]:
				if exp["company"] != None:
					text += normalize_string(exp["company"]) + ' '
				if exp["title"] != None:
					text += normalize_string(exp["title"]) + ' '
				if exp["summary"] != None:
					text += normalize_string(exp["summary"]) + ' '

		if data["summary"] != None:
			text += normalize_string(data["summary"])
			text += ' '

		if data["currentIndustry"] != None:
			text += normalize_string(data["currentIndustry"])
			text += ' '

		cnt += 1
		cur_line = ''
		#for l in label:
		cur_line += ('__label__' + label)
		cur_line += ' '
		cur_line += text
		bunch.append(cur_line + '\n')
		if cnt % bunchsize == 0:
			logging.info("%s \t #lines: %d ", get_time_str(), cnt)
			w.writelines(bunch)
			bunch = []
	w.writelines(bunch)



