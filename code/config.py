# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 18-1-23 下午4:15
# @Author  : wang shen
# @web    : 
# @File    : config.py

import argparse

train_data_file = '../SemEval2015-Task3-English-data/datasets/CQA-QL-train.xml'
devel_data_file = '../SemEval2015-Task3-English-data/datasets/CQA-QL-devel.xml'
test_data_file = '../SemEval2015-Task3-test-with-GOLD/test_task3_English.xml'
test_gold_true_result = '../SemEval2015-Task3-test-with-GOLD/_gold/CQA-QL-test-gold.txt'

train_data_h5 = '../output-data/train_data.h5'
devel_data_h5 = '../output-data/devel_data.h5'
test_data_h5 = '../output-data/test_data.h5'

word_pos_gold_pick = '../output-data/word_pos_gold.pick'
word_vocab_pick = '../output-data/word_vocab.pick'
pos_pick = '../output-data/pos.pick'
gold_pick = '../output-data/gold.pick'

load_model_dir = ''


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--devel_data_file', type=str, default=devel_data_file)
    parser.add_argument('--test_data_file', type=str, default=test_data_file)
    parser.add_argument('--test_gold_true_result', type=str, default=test_gold_true_result)
    parser.add_argument('--train_data_h5', type=str, default=train_data_h5)
    parser.add_argument('--devel_data_h5', type=str, default=devel_data_h5)
    parser.add_argument('--test_data_h5', type=str, default=test_data_h5)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--load_model_dir', type=str, default=load_model_dir)

    parser.add_argument('--word_pos_gold_pick', type=str, default=word_pos_gold_pick)
    parser.add_argument('--word_vocab_pick', type=str, default=word_vocab_pick)
    parser.add_argument('--pos_h5_pick', type=str, default=pos_pick)
    parser.add_argument('--gold_h5_pick', type=str, default=gold_pick)

    parser.add_argument('--word2id', type=dict, default=None)
    parser.add_argument('--pos2id', type=dict, default=None)
    parser.add_argument('--gold2id', type=dict, default=None)
    parser.add_argument('--word_size', type=int, default=None)
    parser.add_argument('--pos_size', type=int, default=None)
    parser.add_argument('--gold_size', type=int, default=None)

    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--max_subject_length', type=int, default=10)
    parser.add_argument('--max_answer_count', type=int, default=143)
    parser.add_argument('--pos_embed_dim', type=int, default=50)
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--lstm_state_size', type=int, default=360)
    parser.add_argument('--lstm_input_size', type=int, default=100)
    parser.add_argument('--lstm_linear_size', type=int, default=50)
    parser.add_argument('--cnn_state_size', type=int, default=100)
    parser.add_argument('--cnn_linear_size', type=int, default=200)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=500)

    return parser.parse_args()

