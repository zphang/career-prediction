from datetime import datetime
import pickle
import heapq
import re

def get_labels(filename, topk, stopwords):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
        for k, v in list(dic.items()):
            if len(k) <= 1 or k in stopwords:
                del dic[k]
    print("total grams:", len(dic), ". getting top ", topk)
    keys = heapq.nlargest(topk, dic, key=dic.get)
    return dict(zip(keys, range(len(keys))))

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s
    
def get_time_str():
    time_str = str(datetime.now())
    time_str = '_'.join(time_str.split())
    time_str = '_'.join(time_str.split('.'))
    time_str = ''.join(time_str.split(':'))
    return time_str

def get_classes(title, labels):
    classes = []
    title_str = normalize_string(title)
    for w in title_str.split():
        if w in labels:
            return w
    return None
