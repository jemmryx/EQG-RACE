# train file
train_src_file = "./race/para-train.txt"
train_trg_file = "./race/tgt-train.txt"
train_ans_file = "./race/ans-train.txt"
train_key_file = "./race/key-train.txt"
# dev file
dev_src_file = "./race/para-dev.txt"
dev_trg_file = "./race/tgt-dev.txt"
dev_ans_file = "./race/ans-dev.txt"
dev_key_file = "./race/key-dev.txt"

# test file
test_src_file = "./race/para-test.txt"
test_trg_file = "./race/tgt-test.txt"
test_ans_file = "./race/ans-test.txt"
test_key_file = "./race/key-test.txt"

# embedding and dictionary file
embedding = "./data/embedding.pkl"
word2idx_file = "./data/word2idx.pkl"

model_path = "./save/para/train_para/18_2.72"
level = "para"
train = True
device = "cuda:0"
use_gpu = True
debug = False
vocab_size = 45000
freeze_embedding = True

num_epochs = 25
max_len = 400
ans_len = 100
key_len = 100
max_seq_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.1
batch_size = 32
dropout = 0.3
max_grad_norm = 5.0

use_tag = True
use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/"
