import unicodedata
import re
import random
import json
import pickle
import numpy as np
import heapq
from gensim import utils
import argparse
from lstm import LSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
from utils import *

def normalize_string_(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z]+", r"", s)
    return s

def get_labels(filename, topk, stopwords):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
        for k, v in list(dic.items()):
            if len(k) <= 1 or k in stopwords:
                del dic[k]
    print("total grams:", len(dic), ". getting top ", topk)
    keys = heapq.nlargest(topk, dic, key=dic.get)
    return dict(zip(keys, range(len(keys))))

def get_vocabulary(filename):
    with open(filename, 'rb') as f:
        voc = pickle.load(f)
        if '<PAD>' not in voc:
            voc.append('<PAD>')
        if '<UNK>' not in voc:
            voc.append('<UNK>')
    return dict(zip(voc, range(len(voc)))), dict(zip(range(len(voc)), voc))

class Corpus(object):
    def __init__(self, fname, labels, vocab, max_sentence_length=300, max_size=500000):
        self.fname = fname
        self.max_size = max_size
        self.max_sentence_length = max_sentence_length
        self.labels = labels
        self.vocab = vocab

    def __iter__(self):
        i = 0
        with utils.smart_open(self.fname) as fin:
            for line in fin:
                data = json.loads(line)
                if data["title"] == '':
                    continue
                i += 1
                if i >= self.max_size:
                    break
                title_str = normalize_string(data["title"])
                out = {}
                for w in title_str.split():
                    if w in self.labels:
                        out["title"] = self.labels[w]
                        # out["original"] = data["title"] + "###" + w
                        break
                if "title" not in out:
                    continue
                out["expertise"] = []
                for w in re.split(' |,', data["expertise"]):
                    w_ = normalize_string_(w)
                    if w_ == '':
                        continue
                    if w_ in self.vocab:
                        out["expertise"].append(self.vocab[w_])
                    else:
                        out["expertise"].append(self.vocab['<UNK>'])
                    if len(out["expertise"]) == self.max_sentence_length:
                        break
                if len(out["expertise"]) == 0:
                    continue
                padding = self.max_sentence_length - len(out["expertise"])
                if padding > 0:
                    out["expertise"].extend([self.vocab['<PAD>']]*padding)
                # out["expertise"] = [normalize_string_(w) for w in re.split(' |,', data["expertise"]) if w != '']
                yield out


# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
        
    return batches

def evaluate(model, testset, lstm):
    model.eval()
    correct = 0
    total = 0
    for vectors, labels in get_batch(dataset, 100, 1000):
        vectors = Variable(torch.from_numpy(np.array(vectors))).cuda()
        labels = torch.from_numpy(np.array(labels))
        if lstm:
            hidden, c_t = model.init_hidden()
            output, hidden = model(vectors, hidden, c_t)
        else:
            hidden = model.init_hidden()
            output, hidden = model(vectors, hidden)
        
        _, predicted = torch.max(output.data, 1)
        batch_size = labels.size(0)
        n_correct = (predicted == labels).sum()
        total += batch_size
        correct += n_correct
        print("batch acc: %f" % (n_correct/batch_size))
      
    return correct / float(total)

def training_loop(model_prefix, batch_size, num_epochs, num_batches, model, loss_, optim, dataset, test_set, lstm=False):
    cnt = 0
    epoch = 0
    # total_batches = int(len(training_set) / batch_size)
    while epoch < num_epochs:
        for vectors, labels in get_batch(dataset, args.batch_size, args.total_size):
            model.train()
            vectors = Variable(torch.from_numpy(np.array(vectors))).cuda() # batch_size, seq_len
            labels = Variable(torch.from_numpy(np.array(labels)))
            # print(vectors.size, labels.size)
            model.zero_grad()
            
            if lstm:
                hidden, c_t = model.init_hidden()
                output, hidden = model(vectors, hidden, c_t)
            else:
                hidden = model.init_hidden()
                output, hidden = model(vectors, hidden)

            lossy = loss_(output, labels)
            lossy.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optim.step()
            
            cnt += 1
            if cnt % 100 == 0:
                torch.save(model.state_dict(), model_prefix + get_time_str() + ".pt")
            # print(cnt)
            print("Time: %s Epoch %i; Step %i; Loss %f; Acc %f" 
                  %(get_time_str(), epoch, cnt, lossy.data[0], evaluate(model, testset, lstm)))
        epoch += 1
        # step += 1

# def iters(iterable, batch_size=10, total=100):
#     for ndx in range(0, total, batch_size):
#         yield iterable[ndx:min(ndx+batch_size, total)]

def get_batch(iterable, batch_size, total):
    vectors = []
    labels = []
    for k in iterable:
        vectors.append(k["expertise"])
        labels.append(k["title"])
        if len(vectors) == batch_size:
            yield vectors, labels
            vectors = []
            labels = []
    if labels and len(labels) == batch_size:
        yield vectors, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default=1000, type=int, 
        help="batch size for training")
    parser.add_argument("-total", "--total_size", default=500000, type=int, 
        help="total dataset size for training")
    # parser.add_argument("-n_batches", "--n_batches", default=10000, type=int, help="training batches per epoch")
    parser.add_argument("-epochs", "--epochs", default=1, type=int, help="training epochs")
    parser.add_argument("-topk_title", "--topk_title", default=5000, type=int, 
        help="topk words appeared in title, used as the #classifications for title")
    parser.add_argument("-training_data", "--training_data", default='json_usa_nyu_09142017.txt',
        help="location of the training data")
    parser.add_argument("-test_data", "--test_data", default='json_usa_nyu_09142017.txt',
        help="location of the test data")
    parser.add_argument("-grams_file", "--grams_file", default='data_title_words_unigram.pickle',
        help="location of the unigram/bigram(extracted from title) file") 
    parser.add_argument("-stopwords_file", "--stopwords_file", default='stopwords.tsv',
        help="stopwords file location")
    parser.add_argument("-vocab_file", "--vocab_file", default='SortedVocab_stripped.pk',
        help="vocabulary file location, size 500000")
    parser.add_argument("-model_prefix", "--model_prefix", default='model_',
        help="model name prefix")
    parser.add_argument("-to_evaluate", "--to_evaluate", action="store_true", default=False,
                      help="to load model and evaluate")
    parser.add_argument("-saved_model", "--saved_model", default='',
                      help="get saved model")

    args = parser.parse_args()
    swfile = args.stopwords_file
    with open(swfile) as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]
    vocab, idx_to_word = get_vocabulary(args.vocab_file)
    
    labels = get_labels(args.grams_file, args.topk_title, stopwords)
    dataset = Corpus(args.training_data, labels, vocab, max_size=args.total_size)
    testset = Corpus(args.test_data, labels, vocab, max_size=1000)
    # for batch in get_batch(dataset, args.batch_size, args.total_size):
    #     # print(batch)
    #     for i in batch:
    #         print(i)
    #         print(len(i['expertise']))
    #     print("\n")

    # hyper-parameters
    vocab_size = len(vocab)
    input_size = vocab_size
    hidden_dim = 256
    embedding_dim = 300
    learning_rate = 0.5
    num_labels = len(labels)
    num_batches = args.total_size//args.batch_size

    rnn = LSTM(vocab_size, embedding_dim, hidden_dim, num_labels, args.batch_size)
    rnn.cuda()
    rnn.init_weights()
    loss = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    
    if args.to_evaluate:
        print("Start loading model %s" % args.saved_model)
        rnn.load_state_dict(torch.load(args.saved_model))
        print("Model loaded")
        acc = evaluate(rnn, testset, lstm=True)
        print("Acc: %f" % acc)
    else:
        print("Start training...")
        training_loop(args.model_prefix, args.batch_size, args.epochs, num_batches, rnn, loss, optimizer, dataset, testset,lstm=True)
        torch.save(rnn.state_dict(), "model_" + get_time_str() + ".pt")
