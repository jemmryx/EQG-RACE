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
    def __init__(self, src_file, trg_file, ans_file, key_file, max_length, word2idx, debug=False):
        self.srcs = []
        self.tags = []
        self.trgs = open(trg_file, "r").readlines()
        self.ans = open(ans_file, "r").readlines()
        self.key = open(key_file, "r").readlines()

        self.max_length = max_length
        self.word2idx = word2idx

        lines = open(src_file, "r").readlines()
        sentence, tags = [], []
        self.entity2idx = {"O": 0, "B": 1, "I": 2, "S": 3}
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                sentence.insert(0, START_TOKEN)
                sentence.append(END_TOKEN)
                self.srcs.append(sentence)

                tags.insert(0, self.entity2idx["O"])
                tags.append(self.entity2idx["O"])
                self.tags.append(tags)
                assert len(sentence) == len(tags)
                sentence, tags = [], []
            else:
                tokens = line.split("\t")
                word, tag = tokens[0], tokens[1]
                sentence.append(word)
                tags.append(self.entity2idx[tag])

        assert len(self.srcs) == len(self.trgs) == len(self.ans) == len(self.tags) == len(self.key), \
            "{} {} {} {} {} must be the same" \
                .format(len(self.srcs), len(self.trgs), len(self.ans), len(self.tags), len(self.key))

        self.num_seqs = len(self.srcs)

        if debug:
            self.srcs = self.srcs[:100]
            self.trgs = self.trgs[:100]
            self.tags = self.tags[:100]
            self.key = self.key[:100]
            self.ans = self.ans[:100]

            self.num_seqs = 100

    def __getitem__(self, index):
        src_seq = self.srcs[index]
        trg_seq = self.trgs[index]
        tag_seq = self.tags[index]
        ans_seq = self.ans[index]
        key_seq = self.key[index]

        tag_seq = torch.Tensor(tag_seq[:self.max_length])
        src_seq, ext_src_seq, oov_lst = self.context2ids(src_seq, self.word2idx)
        trg_seq, ext_trg_seq = self.question2ids(trg_seq, self.word2idx, oov_lst)
        ans_seq, _, _ = self.context2ids(ans_seq, self.word2idx)
        key_seq, _, _ = self.context2ids(key_seq, self.word2idx)

        return src_seq, ext_src_seq, trg_seq, ext_trg_seq, oov_lst, tag_seq, ans_seq, key_seq

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

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst, tag_seqs, ans_seqs, key_seqs = zip(*data)

    src_seqs, src_len = merge(src_seqs)
    ext_src_seqs, _ = merge(ext_src_seqs)
    trg_seqs, trg_len = merge(trg_seqs)
    ext_trg_seqs, _ = merge(ext_trg_seqs)
    tag_seqs, _ = merge(tag_seqs)
    ans_seqs, ans_len = merge(ans_seqs)
    key_seqs, key_len = merge(key_seqs)

    assert src_seqs.size(1) == tag_seqs.size(1), "length of tokens and tags should be equal"

    return src_seqs, ext_src_seqs, src_len, trg_seqs, ext_trg_seqs, trg_len, tag_seqs, oov_lst, ans_seqs, ans_len, key_seqs, key_len


def get_loader(src_file, trg_file, ans_file, key_file, word2idx,
               batch_size, use_tag=False, debug=False, shuffle=False):
    dataset = SQuadDatasetWithTag(src_file, trg_file, ans_file, key_file, config.max_seq_len,
                                  word2idx, debug)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn_tag)

    return dataloader


def make_vocab_from_race(output_file, counter, max_vocab_size):
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3

    for idx, (token, freq) in enumerate(sorted_vocab, start=4):
        if len(word2idx) == max_vocab_size:
            break
        word2idx[token] = idx
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def make_embedding(embedding_file, output_file, word2idx):
    word2embedding = dict()
    lines = io.open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    num_oov = 0
    for word, idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
    return embedding


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


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    # return [token for token in tokens.strip().split()]


def genearl_split(tokens):
    return [token for token in tokens.strip().split()]


def get_truncated_context(context, answer_text, answer_end, parser):
    # get sentences up to the sentence that contains answer span
    doc = parser(context)
    sentences = doc.sentences  # list of Sentence objects
    sents_text = []

    for sentence in sentences:
        sent = []
        for token in sentence.tokens:
            sent.append(token.text)
        sents_text.append(" ".join(sent))
    sentences = sents_text

    stop_idx = -1
    for idx, sentence in enumerate(sentences):
        if answer_text in sentence:
            chars = " ".join(sentences[:idx + 1])
            if len(chars) >= answer_end:
                stop_idx = idx
                break
    if stop_idx == -1:
        print(answer_text)
        print(context)
    truncated_sentences = sentences[:stop_idx + 1]
    truncated_context = " ".join(truncated_sentences).lower()
    return truncated_context


def tokenize(doc, parser):
    words = []
    sentences = parser(doc).sentences
    for sent in sentences:
        toks = sent.tokens
        for token in toks:
            words.append(token.text.lower())
    return words


import itertools


def process_file(file_name):
    counter = defaultdict(lambda: 0)
    examples = list()
    with open(file_name, "r") as f:
        source = json.loads(f.read())

        print(file_name, len(source))
        for sample in tqdm(source):
            article = " ".join(sample['sent']).split()

            key_sent = sample['max_sent'].split()

            answer = sample['answer']

            question = sample['question'].split()

            tag = list(itertools.chain(*sample['tag']))
            assert len(article) == len(tag)

            for token in article + key_sent + answer + question:
                counter[token] += 1

            example = {"context_tokens": article, "key_sent": key_sent, "question": question, "answer": answer,
                       "tag": tag}
            examples.append(example)

    return examples, counter


from nltk.corpus import stopwords
from string import punctuation

stop_word_list = set(stopwords.words('english'))
punctions = set(punctuation)


def keywords_filter(tokens):
    res = []
    for token in tokens:
        if token in stop_word_list or token in punctions:
            continue
        else:
            res.append(token)
    return res


def make_conll_format(examples, src_file, trg_file, ans_file, key_file):
    src_fw = open(src_file, "w")
    trg_fw = open(trg_file, "w")
    ans_fw = open(ans_file, 'w')
    key_fw = open(key_file, 'w')
    print(len(examples))

    for example in tqdm(examples):
        c_tokens = example["context_tokens"]
        copied_tokens = deepcopy(c_tokens)
        q_tokens = example["question"]
        answer = example["answer"]
        tags = example["tag"]
        tags = list(itertools.chain(*tags))
        key_sent = example["key_sent"]

        if config.use_tag:
            for token, tag in zip(copied_tokens, tags):
                src_fw.write(token + "\t" + tag + "\n")
        else:
            for token in copied_tokens:
                src_fw.write(token + "\n")

        src_fw.write("\n")
        question = " ".join(q_tokens)
        trg_fw.write(question + "\n")
        answer = " ".join(answer)
        ans_fw.write(answer + "\n")
        key_sent = " ".join(key_sent)
        key_fw.write(key_sent + "\n")

    src_fw.close()
    trg_fw.close()
    ans_fw.close()
