# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 18-1-23 下午4:45
# @Author  : wang shen
# @web    : 
# @File    : process.py

import xml.etree.ElementTree as ET
import numpy as np
import config
import h5py
import pickle
import nltk
import os
import re

_PAD = '_PAD'
_UNK = '_UNK'

TOKENIZER_RE = re.compile(r'[a-zA-Z0-9,.!?_]+')


def string_clean(string):
    string = string.replace(',', '').replace('.', '').replace('?', '').replace('!', '')
    string = string.replace(';', '').replace(':', ' ').replace('(', '').replace(')', '')
    string = string.replace('-', ' ').replace('_', ' ')
    string = string.replace("'ve", ' have').replace("'s", ' is').replace("'re", ' are').replace("'m", ' am')
    string = string.replace("'d", ' would').replace("'ll", ' will').replace("can't", 'can not')
    string = string.lower().strip()

    words = re.findall(TOKENIZER_RE, string)
    words = [w for w in words if len(w) > 1 or w in ['i', 'a', ',', '.', '!', '?']]

    return words


def pos_tag(string_list):
    r = []
    p = nltk.tag.pos_tag(string_list)
    for k in range(len(p)):
        r.append(p[k][1])

    return r


def get_data2id(data_set):
    data_list = [_PAD]
    result = sorted(list(data_set))
    data_list.extend(list(result))
    data_list += [_UNK]
    data_size = len(data_list)
    data2id = dict(zip(data_list, range(data_size)))
    print('data size :', data_size)

    return data_list, data2id, data_size


def get_max_answer_count(task_path):
    n = 0
    for path in task_path:
        tree = ET.parse(path)
        root = tree.getroot()

        for children in root:
            if children.attrib['QTYPE'] == 'GENERAL':
                if n < len(children) - 2:
                    n = len(children) - 2

    print('question max answer count: ', n)
    return n
#-------------------------------------------------------


def load_word_pos_gold(task_path):
    word_set, pos_set, gold_set = set(), set(), set()

    for path in task_path:
        print('load word pos gold from ', path)
        tree = ET.parse(path)
        root = tree.getroot()

        for children in root:
            if children.attrib['QTYPE'] == 'GENERAL':
                q_subject = string_clean(children[0].text)
                q = string_clean(children[1].text)
                word_set = word_set | set(q_subject + q)
                pos_set = pos_set | set(pos_tag(q_subject + q))

                for k in range(2, len(children)):
                    gold = children[k].attrib['CGOLD']
                    if gold == '?':
                        print(children.attrib['QID'])
                    gold_set.add(gold)

                    answer = string_clean(children[k][1].text)
                    word_set = word_set | set(answer)
                    pos_set = pos_set | set(pos_tag(answer))

    return word_set, pos_set, gold_set


def save(path, data_dict):
    print('saving data2id dict')
    f = open(path, 'wb')
    pickle.dump(data_dict, f)
    f.close()


def load(path):
    print('loading data2id dict')
    f = open(path, 'rb')
    data_dict = pickle.load(f)
    f.close()
    return data_dict


def load_test_gold_result(config):
    f = open(config.test_gold_true_result, 'r')
    print('gold dict: ', config.gold2id)

    result = []
    for line in f:
        line_split = line.split()
        if len(line_split) == 2:
            gold = line_split[1]
        elif len(line_split) == 3:
            gold = line_split[1] + ' ' + line_split[2]

        result.append(config.gold2id[gold])

    return result

#--------------------------------------------------------


def get_answer_feature(q_text, a_text, a_length):
    if a_length == 0:
        return [0] * len(a_text)

    a_feature = []
    for k in range(len(a_text)):
        if a_text[k] in q_text and k < a_length:
            a_feature.append(1)
        else:
            a_feature.append(0)

    return a_feature


def pad_answer(seq_max_size, data2id):
    vector = [data2id[_PAD]] * seq_max_size
    mask = [1] * seq_max_size

    return vector, mask


def pad_seq(seq, seq_max_size, data2id):
    vector = []
    mask = []
    for k in range(seq_max_size):
        if k >= len(seq):
            vector.append(data2id[_PAD])
        elif seq[k] not in data2id.keys():
            vector.append(data2id[_UNK])
        else:
            vector.append(data2id[seq[k]])

        if k >= len(seq):
            mask.append(1)
        else:
            mask.append(0)
    assert len(vector) == len(mask)

    return vector, mask


def save_h5(old_path, new_path, config, flag):
    print('save model data from ', old_path, 'to ', new_path)
    f = h5py.File(new_path, 'w')
    q_subjects, questions, q_pos, answers, a_masks, a_pos, a_features, a_ns, a_labels = [], [], [], [], [], [], [], [], []

    tree = ET.parse(old_path)
    root = tree.getroot()
    for children in root:
        if children.attrib['QTYPE'] == 'GENERAL':
            q_s = string_clean(children[0].text)
            q_s_pad, _ = pad_seq(q_s, config.max_subject_length, config.word2id)

            q = string_clean(children[1].text)
            q_pad, _ = pad_seq(q, config.max_length, config.word2id)
            q_p_pad, _ = pad_seq(pos_tag(q), config.max_length, config.pos2id)

            local_a, local_a_pos, local_a_mask, local_a_features, local_label = [], [], [], [], []
            for k in range(config.max_answer_count):
                if k >= len(children) - 2:
                    local_label.append(config.gold2id[_PAD])

                    a_pad, a_mask = pad_answer(config.max_length, config.word2id)
                    a_p_pad, _ = pad_answer(config.max_length, config.pos2id)
                    a_feature = get_answer_feature(q_pad, a_pad, 0)
                else:
                    if flag != 2:
                        gold = children[k + 2].attrib['CGOLD']
                        local_label.append(config.gold2id[gold])
                    else:
                        local_label.append(config.gold2id[_PAD])

                    a = string_clean(children[k + 2][1].text)
                    a_pad, a_mask = pad_seq(a, config.max_length, config.word2id)
                    a_p_pad, _ = pad_seq(pos_tag(a), config.max_length, config.pos2id)
                    a_feature = get_answer_feature(q_pad, a_pad, len(a))

                local_a.append(a_pad)
                local_a_pos.append(a_p_pad)
                # local_a_mask.append(a_mask)
                local_a_features.append(a_feature)

            q_subjects.append(q_s_pad)
            questions.append(q_pad)
            q_pos.append(q_p_pad)
            answers.append(local_a)
            a_masks.append([1] * (len(children) - 2) + [0] * (config.max_answer_count - len(children) + 2))
            a_pos.append(local_a_pos)
            a_features.append(local_a_features)
            a_ns.append([len(children) - 2])
            a_labels.append(local_label)

    f.create_dataset('q_subjects', data=np.asarray(q_subjects))
    f.create_dataset('questions', data=np.asarray(questions))
    f.create_dataset('q_pos', data=np.asarray(q_pos))
    f.create_dataset('answers', data=np.asarray(answers))
    f.create_dataset('a_masks', data=np.asarray(a_masks))
    f.create_dataset('a_pos', data=np.asarray(a_pos))
    f.create_dataset('a_features', data=np.asarray(a_features))
    f.create_dataset('a_ns', data=np.asarray(a_ns))
    f.create_dataset('a_labels', data=np.asarray(a_labels))

    f.close()

#--------------------------------------------------------


if __name__ == '__main__':
    config = config.get_args()

    task_path = [config.train_data_file, config.devel_data_file, config.test_data_file]
    h5py_path = [config.train_data_h5, config.devel_data_h5, config.test_data_h5]

    print('load data')
    if not os.path.exists(config.word_pos_gold_pick):
        word_set, pos_set, gold_set = load_word_pos_gold(task_path)
        word_list, word2id, word_size = get_data2id(word_set)
        pos_list, pos2id, pos_size = get_data2id(pos_set)
        gold_list, gold2id, gold_size = get_data2id(gold_set)

        data_dict = {'word2id': word2id, 'pos2id': pos2id, 'gold2id': gold2id}
        save(config.word_pos_gold_pick, data_dict)
    else:
        data_dict = load(config.word_pos_gold_pick)
        word2id, pos2id, gold2id = data_dict['word2id'], data_dict['pos2id'], data_dict['gold2id']

    config.word2id = word2id
    config.pos2id = pos2id
    config.gold2id = gold2id

    print('word size: ', len(word2id), '   pos size: ', len(pos2id), '    gold size: ', len(gold2id))

    config.max_answer_count = get_max_answer_count(task_path)
    print('max answer count:', config.max_answer_count)

    # for i in range(len(task_path)):
    #     print('data saving from ', task_path[i], 'to ', h5py_path[i])
    #     save_h5(task_path[i], h5py_path[i], config, i)

    # test
    # load_test_gold_result(config)