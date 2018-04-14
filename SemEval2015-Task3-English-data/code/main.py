# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 18-1-23 下午10:27
# @Author  : wang shen
# @web    : 
# @File    : main.py

import torch.utils.data as data
from code.process import load, load_test_gold_result
from code.config import get_args
from datetime import datetime
from code.baseline import baseline
import numpy as np
from sklearn.metrics import precision_score
from torch.autograd import Variable
import torch
import h5py
import os


class loadDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.n = len(self.file['q_subjects'])

    def __getitem__(self, item):
        q_subject = self.file['q_subjects'][item]
        question = self.file['questions'][item]
        q_p = self.file['q_pos'][item]
        answer = self.file['answers'][item]
        a_mask = self.file['a_masks'][item]
        a_p = self.file['a_pos'][item]
        a_feature = self.file['a_features'][item]
        a_n = self.file['a_ns'][item]
        a_label = self.file['a_labels'][item]

        return q_subject, question, q_p, answer, a_mask, a_p, a_feature, a_n, a_label

    def __len__(self):
        return self.n


def save_model(model, epoch, loss, acc, base_dir):
    model_path = base_dir + '/' + str(epoch) + '_loss' + str(round(loss, 4)) + '_acc' + str(round(acc, 4))
    with open(model_path, 'wb') as f:
        torch.save(model, f)


def train_epoch(model, epoch, train_batchs, opt):
    print('Train epoch :', epoch)
    model.train()

    e_loss, nb = 0.0, 0
    true_label, pred_label = [], []
    for batch_id, (q_subject, question, q_p, answer, a_mask, a_p, a_feature, a_n, a_label) in enumerate(train_batchs):
        nb += 1
        q_s_var = Variable(q_subject.long())
        q_var = Variable(question.long())
        q_p_var = Variable(q_p.long())
        a_var = Variable(answer.long())
        a_m_var = Variable(a_mask.byte())
        a_p_var = Variable(a_p.long())
        a_f_var = Variable(a_feature.byte())
        a_n_var = Variable(a_n.long())
        a_l_var = Variable(a_label.long(), requires_grad=False)

        # q_s, q, q_p, answer, a_m, a_p, a_f, a_n, a_label
        batch_loss = model.get_loss(q_s_var, q_var, q_p_var, a_var, a_m_var, a_p_var, a_f_var, a_n_var, a_l_var)
        batch_tag = model.get_tag(q_s_var, q_var, q_p_var, a_var, a_m_var, a_p_var, a_f_var, a_n_var, a_l_var)

        # batch_label = batch_tag.data.cpu().tolist()
        for n, a_l, p_l in zip(a_n, a_label, batch_tag):
            a_l = a_l[:n[0]]
            true_label.extend(a_l)
            pred_label.extend(p_l.data.cpu().tolist())

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        e_loss += sum(batch_loss.data.cpu().numpy())
        print(true_label, len(true_label))

        print(pred_label, len(pred_label))
        print('-------epoch: ', epoch, '  batch: ', batch_id, '-----batch loss: ', batch_loss.data[0],
              ', macro p: ', precision_score(np.array(true_label), np.array(pred_label), average='macro'))

    e_loss = e_loss / nb
    macro_p = precision_score(true_label, pred_label, average='macro')

    print('----Epoch: ', epoch, ' Train loss: ', e_loss, '  Macro p: ', macro_p)

    return e_loss, macro_p


def train(model, train_batchs, config):
    print('Start Train Model')
    para = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.SGD(para, lr=config.learning_rate)

    acc = 0.9
    for e in range(config.num_epochs):
        l, a = train_epoch(model, e, train_batchs, opt)

        if a >= acc:
            acc = a
            save_model(model, e, l, config.model_file)
    print('End Train Moder')


def train_model(train_batches, config):
    config.model_file = '../model_' + str(datetime.now()).split('.')[0].split()[0]
    if not os.path.exists(config.model_file):
        os.makedirs(config.model_file)

    model = baseline(config)
    train(model, train_batches, config)


def test_model(test_batches, config):
    model = torch.load(config.load_model_dir)

    print('Test')
    true_label = load_test_gold_result(config)
    pred_label = []
    start, end = 0, 0
    for batch_id, (q_subject, question, q_p, answer, a_mask, a_p, a_feature, a_n, a_label) in enumerate(test_batches):
        q_s_var = Variable(q_subject.long())
        q_var = Variable(question.long())
        q_p_var = Variable(q_p.long())
        a_var = Variable(answer.long())
        a_m_var = Variable(a_mask.byte())
        a_p_var = Variable(a_p.long())
        a_f_var = Variable(a_feature.long())
        a_n_var = Variable(a_n.long())
        a_l_var = Variable(a_label.long())

        batch_loss = model.get_loss(q_s_var, q_var, q_p_var, a_var, a_m_var, a_p_var, a_f_var, a_n_var, a_l_var)
        batch_tag = model.get_tag(q_s_var, q_var, q_p_var, a_var, a_m_var, a_p_var, a_f_var, a_n_var, a_l_var)

        local = []
        for tag in batch_tag:
            local += tag
        for n in a_n:
            end += n[0]
        assert len(local) == end - start

        pred_label.extend(local)
        print('macro p: ', precision_score(np.array(true_label[start:end]), np.array(local), average='macro'))
        print('batch loss: ', batch_loss)

        start = end

    print('the all macro p: ', precision_score(np.array(true_label), np.array(pred_label), average='macro'))


if __name__ == '__main__':
    config = get_args()

    train_dataset = loadDataset(config.train_data_h5)
    devel_dataset = loadDataset(config.devel_data_h5)
    test_dataset = loadDataset(config.test_data_h5)

    train_batches = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=config.num_workers,
                                                shuffle=True)
    devel_batches = torch.utils.data.DataLoader(devel_dataset,
                                                batch_size=config.batch_size,
                                                num_workers=config.num_workers,
                                                shuffle=True)
    test_batches = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=False)

    if os.path.exists(config.word_pos_gold_pick):
        data_dict = load(config.word_pos_gold_pick)
        config.word2id, config.pos2id, config.gold2id = data_dict['word2id'], data_dict['pos2id'], data_dict['gold2id']
        config.word_size, config.pos_size, config.gold_size = len(config.word2id), len(config.pos2id), len(config.gold2id)
        print('word size: ', len(config.word2id), ',  pos size : ', len(config.pos2id), ', gold size: ', len(config.gold2id))
    else:
        print('word pos gold file do not exist!!!!')

    print('main -----start train model')
    train_model(train_batches, config)
    print('main -----end train model')

    print('main -----test model')
    # test_model(test_batches, config)
    print('main -----test model')


