# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 18-1-24 上午9:18
# @Author  : wang shen
# @web    : 
# @File    : baseline.py

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch


class baseline(nn.Module):

    def __init__(self, config):
        super(baseline, self).__init__()
        self.word_size = config.word_size
        self.pos_size = config.pos_size
        self.gold_size = config.gold_size
        self.max_length = config.max_length
        self.max_subject_length = config.max_subject_length
        self.word_embed_dim = config.word_embed_dim
        self.pos_embed_dim = config.pos_embed_dim
        self.lstm_input_size = config.lstm_input_size
        self.lstm_state_size = config.lstm_state_size
        self.lstm_linear_size = config.lstm_linear_size
        self.num_workers = config.num_workers
        self.num_layers = config.num_layers
        self.dropout = config.dropout

        self.cnn_state_size = config.cnn_state_size

        self.lookup_word = nn.Embedding(self.word_size, self.word_embed_dim)
        self.lookup_pos = nn.Embedding(self.pos_size, self.pos_embed_dim)
        self.q_s_linear = nn.Linear(self.max_subject_length * self.word_embed_dim, self.word_embed_dim)
        self.input_linear = nn.Linear(300, self.lstm_input_size)
        self.out_linear = nn.Linear(self.lstm_state_size, self.lstm_linear_size)
        self.out_gold_linear = nn.Linear(self.lstm_linear_size, self.gold_size)

        self.q_cnn_layer1 = nn.Conv2d(1, 100, kernel_size=(5, 250), stride=1, padding=0)
        self.q_cnn_pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=1, padding=0)
        self.q_cnn_layer2 = nn.Conv2d(100, 100, kernel_size=(4, 1), stride=1, padding=0)
        self.q_cnn_pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=0)
        self.q_cnn_layer3 = nn.Conv2d(100, 100, kernel_size=(2, 1), stride=1, padding=0)
        self.q_cnn_pool3 = nn.MaxPool2d(kernel_size=(87, 1), stride=1, padding=0)

        self.a_cnn_layer1 = nn.Conv2d(1, 100, kernel_size=(5, 150), stride=1, padding=0)
        self.a_cnn_pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=1, padding=0)
        self.a_cnn_layer2 = nn.Conv2d(100, 100, kernel_size=(4, 1), stride=1, padding=0)
        self.a_cnn_pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=0)
        self.a_cnn_layer3 = nn.Conv2d(100, 100, kernel_size=(2, 1), stride=1, padding=0)
        self.a_cnn_pool3 = nn.MaxPool2d(kernel_size=(87, 1), stride=1, padding=0)

        self.q_cnn_1_layer = nn.Conv2d(1, 100, kernel_size=(5, 250), stride=1, padding=0)
        self.q_cnn_1_pool = nn.MaxPool2d(kernel_size=(96, 1), stride=1, padding=0)

        self.a_cnn_1_layer = nn.Conv2d(1, 100, kernel_size=(5, 150), stride=1, padding=0)
        self.a_cnn_1_pool = nn.MaxPool2d(kernel_size=(96, 1), stride=1, padding=0)
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_state_size // 2, self.num_layers, dropout=self.dropout, bidirectional=True)

        self.loss = nn.NLLLoss()

    def init_hidden(self, num_layers, batch_size, hidden_size):
        h0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        c0 = Variable(torch.zeros(num_layers, batch_size, hidden_size))

        return (h0, c0)

    def q_cnn_3_layers(self, input):
        input = torch.unsqueeze(input, 1)

        layer1_out = self.q_cnn_layer1(input)
        pool1_out = self.q_cnn_pool1(layer1_out)
        layer2_out = self.q_cnn_layer2(pool1_out)
        pool2_out = self.q_cnn_pool2(layer2_out)
        layer3_out = self.q_cnn_layer3(pool2_out)
        pool3_out = self.q_cnn_pool3(layer3_out)

        out = pool3_out.view(input.size(0), -1)

        return out

    def q_cnn_1_layers(self, input):
        input = torch.unsqueeze(input, 1)

        out = self.q_cnn_1_layer(input)
        out = self.q_cnn_1_pool(out)

        out = out.view(input.size(0), -1)

        return out

    def question_cnn(self, q_s, q, q_p):
        q_embed = self.lookup_word(q)
        q_s_embed = self.lookup_word(q_s)
        q_p_embed = self.lookup_pos(q_p)

        q_s_embed = self.q_s_linear(q_s_embed.view(q.size(0), -1))
        q_s_embed = q_s_embed.expand(self.max_length, *q_s_embed.size())
        q_s_embed = q_s_embed.transpose(0, 1).contiguous()

        q_input = torch.cat([q_embed, q_s_embed, q_p_embed], -1)
        q_out = self.q_cnn_1_layers(q_input)

        return q_out

    def answer_cnn(self, answer, a_p):
        a_list = []

        for a_e, a_p_e in zip(answer, a_p):
            a_embed = self.lookup_word(a_e)
            a_p_embed = self.lookup_pos(a_p_e)

            a_input = torch.cat([a_embed, a_p_embed], -1)
            a_input = torch.unsqueeze(a_input, 1)

            # three layers cnn
            # layer1_out = self.a_cnn_layer1(a_input)
            # pool1_out = self.a_cnn_pool1(layer1_out)
            # layer2_out = self.a_cnn_layer2(pool1_out)
            # pool2_out = self.a_cnn_pool2(layer2_out)
            # layer3_out = self.a_cnn_layer3(pool2_out)
            # out = self.a_cnn_pool3(layer3_out)

            # one layer cnn
            out = self.a_cnn_1_layer(a_input)
            out = self.a_cnn_1_pool(out)

            out = out.view(a_e.size(0), -1)
            a_list.append(out)

        a_out = torch.cat(a_list, 0).view(answer.size()[0], *a_list[0].size())

        return a_out

    def get_pack_inputs(self, x, x_mask):
        l = x_mask.data.eq(0).long().sum(1).squeeze()
        _, id_sort = torch.sort(l, dim=0, descending=True)
        _, id_unsort = torch.sort(id_sort, dim=0)

        l = list(l[id_sort])
        x = x.index_select(0, Variable(id_sort))
        x = x.transpose(0, 1).contiguous()
        input = nn.utils.rnn.pack_padded_sequence(x, l)

        return input, Variable(id_unsort)

    def get_pad_outputs(self, x, x_mask, id_unsort):
        out = nn.utils.rnn.pad_packed_sequence(x)[0]

        out = out.transpose(0, 1).contiguous()
        out = out.index_select(0, id_unsort)

        if out.size(1) != x_mask.size(1):
            padding = torch.zeros(out.size(0),
                                  x_mask.size(1) - out.size(1),
                                  out.size(2)).type(out.data.type())
            out = torch.cat([out, Variable(padding)], 1)

        return out

    def get_lstm(self, q_s, q, q_p, answer, a_m, a_p, a_f):
        batch_size = q_s.size()[0]
        q_cnn_output = self.question_cnn(q_s, q, q_p)
        # print('q cnn out size: ', q_cnn_output.size())

        # print('max answer count: ', answer.size()[1])
        a_cnn_output = self.answer_cnn(answer, a_p)
        # print('a cnn out size: ', a_cnn_output.size())

        q_cnn_output = q_cnn_output.expand(a_cnn_output.size()[1], *q_cnn_output.size())
        q_cnn_output = q_cnn_output.transpose(0, 1).contiguous()
        # print('q cnn out size: ', q_cnn_output.size())

        lstm_input = torch.cat([a_cnn_output, q_cnn_output, a_f.float()], -1)
        lstm_input = self.input_linear(lstm_input)
        # print('input size: ', lstm_input.size())
        # print('a mask size: ', a_m.size())

        input, id_unsort = self.get_pack_inputs(lstm_input, a_m)
        init_hidden = self.init_hidden(self.num_layers * 2, batch_size, self.lstm_state_size // 2)
        out, _ = self.lstm(input, init_hidden)
        out = self.get_pad_outputs(out, a_m, id_unsort)
        out = self.out_linear(out)

        return out

    def forward(self, q_s, q, q_p, answer, a_m, a_p, a_f):
        network_lstm = self.get_lstm(q_s, q, q_p, answer, a_m, a_p, a_f)

        s_list = []
        for t in network_lstm:
            tag = self.out_gold_linear(t)
            tag_scores = F.log_softmax(tag, -1)
            s_list.append(tag_scores)

        s = torch.cat(s_list, 0).view(len(s_list), *s_list[0].size())

        return s

    def get_loss(self, q_s, q, q_p, answer, a_m, a_p, a_f, a_n, a_label):
        scores = self.forward(q_s, q, q_p, answer, a_m, a_p, a_f)

        l_list = []
        a_n = a_n.data.cpu().tolist()
        for n, s, l in zip(a_n, scores, a_label):
            loss = self.loss(s[:n[0]], l[:n[0]])
            l_list.append(loss)

        batch_s = torch.mean(torch.cat(l_list, -1))

        return batch_s

    def get_tag(self, q_s, q, q_p, answer, a_m, a_p, a_f, a_n, a_label):
        scores = self.forward(q_s, q, q_p, answer, a_m, a_p, a_f)
        scores, idx = torch.max(scores, -1)

        a_n = a_n.data.cpu().tolist()
        tag = []
        # print('a n: ', a_n)
        # print('idx: ', idx)
        for n, index in zip(a_n, idx):
            tag.append(index[:n[0]])

        # tag = torch.cat(tag, 0)
        # print('tag size: ', tag.size())
        return tag






