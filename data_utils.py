import io
import itertools
import json
import pickle
import time
from collections import defaultdict
from copy import deepcopy
from string import punctuation

import config
import nltk
import numpy as np
import torch
import torch.utils.data as data
from nltk.corpus import stopwords
from tqdm import tqdm

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


class SQuadDatasetWithTag(data.Dataset):
    def __init__(self, file, adj_file, max_length, word2idx, debug=False):
        self.passages = []
        self.tags = []
        self.questions = []
        self.answers = []
        self.keys = []
        self.adjs = []

        self.max_length = max_length
        self.word2idx = word2idx
        self.tag2idx = {"O": 0, "B": 1, "I": 2, "S": 3}

        with open(file, 'r')as r:
            lines = r.readlines()
            for line in lines:
                line = line.strip().split("\t")
                passage = line[0].strip().split()
                tag = line[1].strip().split()
                key = line[2].strip().split()
                ans = line[3].strip().split()
                ques = line[4].strip()
                self.passages.append(passage)
                self.tags.append([self.tag2idx[t] for t in tag])
                self.keys.append(key)
                self.answers.append(ans)
                self.questions.append(ques)
        with open(adj_file, 'r')as rr:
            r = rr.readlines()
            print("loading dependency tree....")
            for adj in tqdm(r):
                rows = adj.split("\t")
                new = np.array([r.split() for r in rows], dtype=np.float32)
                if new.shape[0] > config.max_seq_len:
                    temp = np.zeros([config.max_seq_len, config.max_seq_len], dtype=np.float32)
                    temp[:config.max_seq_len - 1, :config.max_seq_len - 1] = new[:config.max_seq_len - 1,
                                                                             :config.max_seq_len - 1]
                    temp[-1, -1] = new[-1, -1]
                    new = temp
                self.adjs.append(new)

        assert len(self.passages) == len(self.tags) == len(self.answers) == len(self.questions) == len(
            self.keys) == len(self.adjs), \
            "{} {} {} {} {} {} must be the same" \
                .format(len(self.passages), len(self.tags), len(self.answers), len(self.questions), len(self.keys),
                        len(self.adjs))

        self.num_seqs = len(self.passages)

        if debug:
            self.passages = self.passages[:100]
            self.tags = self.tags[:100]
            self.questions = self.questions[:100]
            self.answers = self.answers[:100]
            self.keys = self.keys[:100]
            self.adjs = self.adjs[:100]
            self.num_seqs = 100

    def __getitem__(self, index):

        passage = self.passages[index]
        tag = self.tags[index]
        question = self.questions[index]
        answer = self.answers[index]
        key = self.keys[index]
        adj = self.adjs[index]

        tag_seq = torch.Tensor(tag[:self.max_length])
        src_seq, ext_src_seq, oov_lst = self.context2ids(passage, self.word2idx)
        trg_seq, ext_trg_seq = self.question2ids(question, self.word2idx, oov_lst)
        ans_seq, _, _ = self.context2ids(answer, self.word2idx)
        key_seq, _, _ = self.context2ids(key, self.word2idx)
        adj = np.array(adj)

        return src_seq, ext_src_seq, trg_seq, ext_trg_seq, oov_lst, tag_seq, ans_seq, key_seq, adj

    def __len__(self):
        return self.num_seqs

    def context2ids(self, tokens, word2idx):
        ids = list()
        extended_ids = list()
        oov_lst = list()
        # START and END token is already in tokens lst
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            if len(ids) == self.max_length:
                break

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids, oov_lst

    def question2ids(self, sequence, word2idx, oov_lst):
        ids = list()
        extended_ids = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token in oov_lst:
                    extended_ids.append(len(word2idx) + oov_lst.index(token))
                else:
                    extended_ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids


def collate_fn_tag(data):
    def merge(sequences):
        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_adj(adjs):
        numbers = len(adjs)
        lengths = [adj.shape[0] for adj in adjs]
        max_len = min(max(lengths), config.max_seq_len)
        padded_adjs = torch.zeros(numbers, max_len, max_len).type(torch.FloatTensor)
        for i, adj in enumerate(adjs):
            end = lengths[i]
            end = min(end, config.max_seq_len)
            padded_adjs[i, :end, :end] = torch.from_numpy(adj[:end, :end])
        return padded_adjs

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst, tag_seqs, ans_seqs, key_seqs, adjs = zip(*data)

    src_seqs, src_len = merge(src_seqs)
    ext_src_seqs, _ = merge(ext_src_seqs)
    trg_seqs, trg_len = merge(trg_seqs)
    ext_trg_seqs, _ = merge(ext_trg_seqs)
    tag_seqs, _ = merge(tag_seqs)
    ans_seqs, ans_len = merge(ans_seqs)
    key_seqs, key_len = merge(key_seqs)
    adjs = merge_adj(adjs)

    assert src_seqs.size(1) == tag_seqs.size(1), "length of tokens and tags should be equal"

    return src_seqs, ext_src_seqs, src_len, trg_seqs, ext_trg_seqs, trg_len, \
           tag_seqs, oov_lst, ans_seqs, ans_len, key_seqs, key_len, adjs


def get_loader(file, adj_file, word2idx,
               batch_size, use_tag=False, debug=False, shuffle=False):


    dataset = SQuadDatasetWithTag(file, adj_file, config.max_seq_len,
                                  word2idx, debug)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn_tag)

    return dataloader


def time_since(t):
    """ Function for time. """
    return time.time() - t


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)


def outputids2words(id_list, idx2word, article_oovs=None):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param article_oovs: list of oov words
    :return: list of words
    """
    words = []
    for idx in id_list:
        try:
            word = idx2word[idx]
        except KeyError:
            if article_oovs is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = article_oovs[article_oov_idx]
                except IndexError:
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ID]
        words.append(word)

    return words
