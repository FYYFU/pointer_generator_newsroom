import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from beam_search import *
from model.Model import Model

import glob
from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_utils import *
from rouge import Rouge

import argparse


class Evaluate(object):

    def __init__(self, data_path, opt, batch_size=config.batch_size):

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval', batch_size=batch_size,
                               single_pass=True)

        self.opt =opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = torch.load(os.path.join(config.save_model_path,
                                             self.opt.load_model))
        # 加载在train中保存得模型
        self.model.load_state_dict(checkpoint['model_dict'])

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents,
                                 loadfile):

        # 这里可能会存在一点问题，debug得时候剋注意一下
        filename = 'test_' + loadfile.split('.')[:-1] + '.txt'

        with open(os.path.join('data', filename), 'w') as f:
            for i in range(len(decoded_sents)):
                f.write('article' + article_sents[i] + '\n')
                f.write('reference:' + ref_sents[i] + '\n')
                f.write('decoder:' + decoded_sents[i] + '\n')

    def evaluate_batch(self, print_sents =False):

        self.setup_valid()
        batch = self.batcher.next_batch()

        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)

        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()

        batch_number = 0

        while batch is not None:

            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, \
                extra_zeros, ct_e = get_enc_data(batch)

            with torch.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            with torch.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e,
                                       extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                # 返回的是一个 单词列表。
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])

                if len(decoded_words) < 2:
                    decoded_words = 'xxx'
                else:
                    decoded_words = ' '.join(decoded_words)

                decoded_sents.append(decoded_words)
                summary = batch.original_summarys[i]
                article = batch.original_articles[i]
                ref_sents.append(summary)
                article_sents.append(article)

            batch = self.batcher.next_batch()
            batch_number += 1

            if batch_number < 100:
                continue
            else:
                break

        load_file = self.opt.load_model

        if print_sents:
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents, avg=True)

        if self.opt.task == 'test':
            print(load_file, 'scores:', scores)
            sys.stdout.flush()
        else:
            rouge_l = scores['rouge-l']['f']
            print(load_file, 'rouge-l:', '%.4f' % rouge_l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='validate', choices=['validate', 'test'])
    parser.add_argument('--start_from', type=str, default='0020000.tar')
    parser.add_argument('--load_model', type=str, default=None)

    opt = parser.parse_args()

    if opt.task == 'validate':
        # 这一部分代码对应的应该是一个文件
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx: ]

        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config.valid_data_path, opt)
            eval_processor.evaluate_batch()
    else: # test
        print('task:', opt.task)
        print('load_model:', opt.load_model)

        file_list = glob.glob("/home/yuf/pointer_generator/data/*")
        for file in file_list:
            print(file)
            sys.stdout.flush()
            eval_processor = Evaluate(file, opt)
            eval_processor.evaluate_batch()







