import os
import pickle
import time
import warnings

import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2seq

warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model_path=None, level="para"):

        train_file = config.train_file
        train_adj = config.train_adj_file

        dev_file = config.dev_file
        dev_adj = config.dev_adj_file

        # load dictionary and embedding file
        with open(config.embedding, "rb") as f:
            embedding = pickle.load(f, encoding='iso-8859-1')
            device = torch.device(config.device)
            embedding = torch.Tensor(embedding).to(device)
        with open(config.word2idx_file, "rb") as f:
            word2idx = pickle.load(f, encoding='iso-8859-1')

        # train, dev loader
        print("load train data")
        self.train_loader = get_loader(train_file, train_adj,
                                       word2idx,
                                       use_tag=config.use_tag,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        print("load dev data")
        self.dev_loader = get_loader(dev_file, dev_adj,
                                     word2idx,
                                     use_tag=config.use_tag,
                                     batch_size=128,
                                     debug=config.debug)

        train_dir = os.path.join("./save", config.level)
        self.model_dir = os.path.join(train_dir, "train_%s" % config.level)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = Seq2seq(embedding, model_path=model_path)
        params = list(self.model.encoder.parameters()) \
                 + list(self.model.decoder.parameters())

        self.lr = config.lr
        self.optim = optim.SGD(params, self.lr, momentum=0.8)
        # self.optim = optim.Adam(params)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def save_model(self, loss, epoch):
        state_dict = {
            "epoch": epoch,
            "current_loss": loss,
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict()
        }
        loss = round(loss, 2)
        model_save_path = os.path.join(self.model_dir, str(epoch) + "_" + str(loss))
        torch.save(state_dict, model_save_path)

    def train(self):
        batch_num = len(self.train_loader)
        print("Epoch :", config.num_epochs)
        self.model.train_mode()
        best_loss = 1e10
        for epoch in range(1, config.num_epochs + 1):
            print("epoch {}/{} :".format(epoch, config.num_epochs), end="\r")
            start = time.time()
            # halving the learning rate after epoch 8
            if epoch >= 8 and epoch % 2 == 0:
                self.lr *= 0.5
                state_dict = self.optim.state_dict()
                for param_group in state_dict["param_groups"]:
                    param_group["lr"] = self.lr
                self.optim.load_state_dict(state_dict)

            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss = self.step(train_data)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
                self.optim.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")

            val_loss = self.evaluate(msg)
            if val_loss <= best_loss:
                best_loss = val_loss
                self.save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def mask_matrix(self, m):
        mm = []
        for i, seq in enumerate(m):
            mm.append([])
            for token in seq:
                if token > 0:
                    mm[i].append(1)
                else:
                    mm[i].append(0)
        return mm

    def step(self, train_data):

        src_seq, ext_src_seq, src_len, trg_seq, ext_trg_seq, trg_len, tag_seq, _, ans_seq, ans_len, key_seq, key_len, adjs = train_data

        src_len = torch.LongTensor(src_len)
        ans_len = torch.LongTensor(ans_len)
        key_len = torch.LongTensor(key_len)
        enc_zeros = torch.zeros_like(src_seq)
        ans_zeros = torch.zeros_like(ans_seq)
        key_zeros = torch.zeros_like(key_seq)
        #enc_mask = torch.BoolTensor(src_seq == enc_zeros)
        enc_mask=(src_seq == 0).byte()


        adjs = torch.FloatTensor(adjs)
        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            ext_src_seq = ext_src_seq.to(config.device)
            src_len = src_len.to(config.device)
            trg_seq = trg_seq.to(config.device)
            ext_trg_seq = ext_trg_seq.to(config.device)
            enc_mask = enc_mask.to(config.device)

            ans_seq = ans_seq.to(config.device)
            ans_len = ans_len.to(config.device)

            key_seq = key_seq.to(config.device)
            key_len = key_len.to(config.device)
            adjs = adjs.to(config.device)

            if config.use_tag:
                tag_seq = tag_seq.to(config.device)
            else:
                tag_seq = None

        enc_outputs, enc_states = self.model.encoder(src_seq, src_len, tag_seq, ans_seq, ans_len, key_seq, key_len,
                                                     adjs)
        sos_trg = trg_seq[:, :-1]
        eos_trg = trg_seq[:, 1:]

        if config.use_pointer:
            eos_trg = ext_trg_seq[:, 1:]
        logits = self.model.decoder(sos_trg, ext_src_seq, enc_states, enc_outputs, enc_mask)
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)

        loss = self.criterion(preds, targets)
        return loss

    def evaluate(self, msg):
        self.model.eval_mode()
        num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in enumerate(self.dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
                print(msg2, end="\r")
        # go back to train mode
        self.model.train_mode()
        val_loss = np.mean(val_losses)

        return val_loss
