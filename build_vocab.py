'''
build vocabulary from dataset
'''

from nltk.tokenize import RegexpTokenizer
import argparse
import pickle
from datetime import datetime
import json
import logging
from collections import Counter
import re
from os.path import join as pjoin

def get_time_str():
	time_str = str(datetime.now())
	time_str = '_'.join(time_str.split())
	time_str = '_'.join(time_str.split('.'))
	time_str = ''.join(time_str.split(':'))
	return time_str

tokenizer = RegexpTokenizer('[a-z]\w+')


parser = argparse.ArgumentParser()
parser.add_argument('-corpus', '--corpus', type=str, default='')
parser.add_argument('-output', '--output', type=str, default='')
args = parser.parse_args()
log_file = pjoin('build_vocab_' + get_time_str() + '.log')
logging.basicConfig(filename=log_file,level=logging.DEBUG)
vocab = Counter()
cnt = 0
max_expertise = 0
max_experience = 0
max_certification = 0
max_education = 0
max_awards = 0
max_summary = 0
max_industry = 0
max_language = 0
with open(args.corpus, "r") as r:
	for line in r:
		data = json.loads(line)
		if data["title"] == "":
			continue
		# words = tokenizer.tokenize(s.lower())
		if data["expertise"] != "":
			expertise = [s.lower() for s in re.split(' |,', data["expertise"])]
			vocab.update(expertise)
			max_expertise = max(max_expertise, len(expertise))

		if data["educations"] != None:
			education = []
			for edu in data["educations"]:
				if edu["campus"] != None:
					education += tokenizer.tokenize(edu["campus"].lower())
				if edu["major"] != None:
					education += tokenizer.tokenize(edu["major"].lower())
				if edu["specialization"] != None:
					education += tokenizer.tokenize(edu["specialization"].lower())
			vocab.update(education)
			max_education = max(max_education, len(education))

		if data["awards"] != None:
			awards = []
			for a in data["awards"]:
				awards += tokenizer.tokenize(a.lower())
			vocab.update(awards)
			max_awards = max(max_awards, len(awards))

		if data["certifications"] != None:
			certs = []
			for cert in data["certifications"]:
				if cert["title"] != None:
					certs += tokenizer.tokenize(cert["title"].lower())
				if cert["description"] != None:
					certs += tokenizer.tokenize(cert["description"].lower())
			max_certification = max(max_certification, len(certs))
			vocab.update(certs)

		if data["languages"] != None:
			languages = [l.lower() for l in data["languages"]]
			vocab.update(languages)
			max_language = max(max_language, len(languages))

		if data["experiences"] != None:
			experiences = []
			for exp in data["experiences"]:
				if exp["company"] != None:
					experiences += tokenizer.tokenize(exp["company"].lower())
				if exp["title"] != None:
					experiences += tokenizer.tokenize(exp["title"].lower())
				if exp["summary"] != None:
					experiences += tokenizer.tokenize(exp["summary"].lower())
			vocab.update(experiences)
			max_experience = max(max_experience, len(experiences))
		
		if data["summary"] != None:
			summary = tokenizer.tokenize(data["summary"].lower())
			vocab.update(summary)
			max_summary = max(max_summary, len(summary))

		if data["currentIndustry"] != None:
			industry = tokenizer.tokenize(data["currentIndustry"].lower())
			max_industry = max(max_industry, len(industry))
			vocab.update(industry)
		
		cnt += 1
		if cnt % 10000 == 0:
			logging.info("%s \t #lines: %d expert: %d exp: %d cert: %d edu: %d awar: %d indus: %d lan: %d sum: %d", 
				get_time_str(), cnt, max_expertise, max_experience, max_certification, 
				max_education, max_awards, max_industry, max_language, max_summary)

with open(args.output, 'wb') as out:
	pickle.dump(vocab, out)

print(max_expertise, max_experience, max_certification, max_education, max_awards, 
	max_industry, max_language, max_summary)
print(len(vocab))
