import sys

import config
from data_utils import make_embedding, make_vocab_from_race, \
    process_file, make_conll_format


def make_para_dataset():
    embedding_file = "./glove.840B.300d.txt"#need to download
    embedding = "./embedding.pkl"
    src_word2idx_file = "./word2idx.pkl"

    train_race = "train.json"
    dev_race = "dev.json"
    test_race = "test.json"

    train_src_file = "../race/para-train.txt"
    train_trg_file = "../race/tgt-train.txt"
    train_ans_file = "../race/ans-train.txt"
    train_key_file = "../race/key-train.txt"

    dev_src_file = "../race/para-dev.txt"
    dev_trg_file = "../race/tgt-dev.txt"
    dev_ans_file = "../race/ans-dev.txt"
    dev_key_file = "../race/key-dev.txt"


    test_src_file = "../race/para-test.txt"
    test_trg_file = "../race/tgt-test.txt"
    test_ans_file = "../race/ans-test.txt"
    test_key_file = "../race/key-test.txt"


    train_examples, counter = process_file(train_race)
    dev_examples, _ = process_file(dev_race)
    test_examples, _ = process_file(test_race)

    word2idx = make_vocab_from_race(src_word2idx_file, counter, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)

    print("train examples:\n")
    make_conll_format(train_examples, train_src_file, train_trg_file, train_ans_file, train_key_file)
    print("dev examples:\n")
    make_conll_format(dev_examples, dev_src_file, dev_trg_file, dev_ans_file,dev_key_file)
    print("test examples:\n")
    make_conll_format(test_examples, test_src_file, test_trg_file, test_ans_file,test_key_file)


if __name__ == "__main__":
    make_para_dataset()
