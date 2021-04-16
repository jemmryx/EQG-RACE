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
import os
import numpy as np
import torch
import torch.utils.data as data
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.parse.stanford import StanfordDependencyParser

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3
os.environ['STANFORD_PARSER'] = './stanford_parser/stanford-parser.jar'
os.environ[
    'STANFORD_MODELS'] = './stanford_parser/stanford-parser-3.6.0-models.jar'

parser = StanfordDependencyParser(
    model_path="./stanford_parser/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")



def preprocess():
    src_files = [
        "./race/para-train.txt",
        "./race/para-dev.txt",
        "./race/para-test.txt"]

    trg_files = [
        "./race/tgt-train.txt",
        "./race/tgt-dev.txt",
        "./race/tgt-test.txt"]
    ans_files = [
        "./race/ans-train.txt",
        "./race/ans-dev.txt",
        "./race/ans-test.txt"]

    key_files = [
        "./race/key-train.txt",
        "./race/key-dev.txt",
        "./race/key-test.txt"]
    out_files = [
        "./race/train.txt",
        "./race/dev.txt",
        "./race/test.txt"]
    for src_file, trg_file, ans_file, key_file, out_file in zip(src_files, trg_files, ans_files, key_files, out_files):

        srcs = []
        tags = []
        trgs = open(trg_file, "r").readlines()
        ans = open(ans_file, "r").readlines()
        key = open(key_file, "r").readlines()
        max_length = config.max_seq_len

        lines = open(src_file, "r").readlines()
        sentence, tag = [], []
        for line in tqdm(lines):
            line = line.strip()
            if len(line) == 0:
                sentence.insert(0, START_TOKEN)
                tag.insert(0, "O")

                if len(sentence) > max_length:
                    sentence = sentence[:max_length]
                    tag = tag[:max_length]

                sentence.append(END_TOKEN)
                tag.append("O")
                srcs.append(sentence)
                tags.append(tag)
                assert len(sentence) == len(tag)
                sentence, tag = [], []
            else:
                tokens = line.split("\t")
                word, word_tag = tokens[0], tokens[1]
                sentence.append(word)
                tag.append(word_tag)

        assert len(srcs) == len(trgs) == len(ans) == len(tags) == len(key), \
            "{} {} {} {} {} must be the same" \
                .format(len(srcs), len(trgs), len(ans), len(tags), len(key))
        with open(out_file, 'w')as w:
            for sent_, tag_, key_, ans_, ques_ in zip(srcs, tags, key, ans, trgs):
                w.write(" ".join(sent_) + "\t" + " ".join(
                    tag_) + "\t" + key_.strip() + "\t" + ans_.strip() + "\t" + ques_.strip() + "\n")


def get_dep_adj(passage, tag):
    map_passage = {}
    word_passage = passage.split()
    tags = tag.split()
    assert len(word_passage) == len(tags)
    for position, word in enumerate(word_passage):
        map_passage[position] = word
    adj = np.zeros([len(word_passage), len(word_passage)])
    str_passage = " ".join(word_passage)
    sentences = str_passage.replace(".", "#").replace("!", "#").replace("?", "#").split("#")
    start_position = 0
    end_position = 0

    for sent in sentences:
        end_position += len(sent.split())
        if end_position > len(word_passage):
            end_position = len(word_passage)

        flag = tags[start_position:end_position]
        if not ("I" in flag or "B" in flag or "S" in flag):
            start_position = end_position + 1
            adj[end_position - 1][min(start_position, len(word_passage) - 1)] = \
                adj[min(start_position, len(word_passage) - 1)][
                    end_position - 1] = 1
            end_position = start_position
            continue

        try:
            res = list(parser.raw_parse(sent))
        except:
            print("one error occur")
            continue
        for row in res[0].triples():
            word1 = row[0][0]
            word2 = row[2][0]
            pos1 = 0
            pos2 = 0
            for i in range(start_position, end_position):
                if map_passage[i] == word1:
                    pos1 = i
                if map_passage[i] == word2:
                    pos2 = i
            adj[pos1][pos2] = 1
        start_position = end_position + 1
        adj[end_position - 1][min(start_position, len(word_passage) - 1)] = \
            adj[min(start_position, len(word_passage) - 1)][
                end_position - 1] = 1
        end_position = start_position
    for i in range(adj.shape[0]):
        adj[i][i] = 1
    return adj


def dependency():
    in_files = [
        "./race/train.txt",
        "./race/dev.txt",
        "./race/test.txt"]
    out_files = [
        "./race/train.adj",
        "./race/dev.adj",
        "./race/test.adj"
    ]
    for file, out in zip(in_files, out_files):
        print(file)
        with open(file, 'r')as r, open(out, 'w')as w:
            lines = r.readlines()
            for line in tqdm(lines):
                sample = line.split("\t")
                passage = sample[0]
                tag = sample[1]
                adj = get_dep_adj(passage, tag).tolist()
                str_adj = []
                for row in adj:
                    temp_str_adj = []
                    for num in row:
                        temp_str_adj.append(str(num))
                    str_adj.append(temp_str_adj)
                w.write("\t".join([" ".join(row) for row in str_adj]) + "\n")


dependency()
