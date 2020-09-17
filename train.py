import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import torch.nn.functional as F
from model.Model import Model

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_utils import *
from numpy import random
import argparse

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

"""
需要vocab file的存储位置。
"""

class Train(object):
    def __init__(self, opt):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)

        self.opt = opt
        self.start_id = self.vocab.word2id(data.START_DECODING)
        self.end_id = self.vocab.word2id(data.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)

        time.sleep(5)

    def save_model(self, iter):
        save_path = config.save_model_path + "%07d.tar" % iter
        torch.save({
            'iter': iter + 1,
            'model_dict': self.model.state_dict(),
            'training_dict': self.trainer.state_dict()
        }, save_path)

    def setup_train(self):
        self.model = Model()
        self.model = get_cuda(self.model)

        self.trainer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        start_iter = 0

        if self.opt.load_model is not None:
            load_model_path = os.path.join(config.save_model_path, self.opt.load_model)
            checkpoint = torch.load(load_model_path)
            start_iter = checkpoint['iter']
            self.model.load_state_dict(checkpoint['model_dict'])
            self.trainer.load_state_dict(checkpoint['trainer_dict'])
            print("load model at" + load_model_path)

        if self.opt.new_lr is not None:
            self.trainer = torch.optim.Adam(self.model.parameters(), lr=self.opt.new_lr)
            # for params in self.traine
            # .param_groups:
            #     params['lr'] = self.opt.new_lr

        return start_iter


    def train_batch_MLE(self, enc_out, enc_hidden, enc_padding_mask, ct_e,
                        extra_zeros, enc_batch_extend_vocab, batch):
        '''
        以0.25的概率使用生成token来作为输入，0.75的概率以ground-truth label作为输入。

        输入：
        enc_out: encoder的每个time step的输出。
            [batch_size, max_seq_len, 2 * hidden_dim]

        enc_hidden: encoder最后的单元的隐藏状态和记忆状态。 (h, c)
            [batch_size, hidden_dim]

        enc_padding_mask: 对encoder的输入区分padding部分和确切的输入部分。
            因为输入的时候是按照最长的单元的长度来设定的，所以在形成batch的时候进行了padding操作。
            [batch_size, max_seq_len]. 0代表填充，1代表没有填充。

        ct_e: decoder的time step对encoder进行attention操作得到的向量。
            [batch_size, 2 * hidden_dim]. 随着time step而不断的变化的。

        extra_zeros:存储oovs。
            [batch_size, max_art_oovs]

        enc_batch_extend_vocab: 输入的batch，并且里面的各个article的oov都使用了对应的temperatual oov id来表示。
            [batch_size, max_seq_len]

        batch: 输入的batch, 类 Batch的对象。
        '''

        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)

        step_losses = []

        h_t = (enc_hidden[0], enc_hidden[1])

        x_t = get_cuda(torch.LongTensor(len(enc_out)).fill_(self.start_id))

        prev_s = None
        sum_temporal_srcs = None

        for t in range(min(max_dec_len, config.max_dec_steps)):
            # 对于batch中的每个article，随机生成一个数字，
            # 从而得到对应的article是否使用ground-truth label。得到0/1
            use_ground_truth = get_cuda((torch.rand(len(enc_out)) > 0.25)).long()

            x_t = use_ground_truth * dec_batch[:, t] + (1 - use_ground_truth) * x_t
            # 这里我觉得有一点不太对，
            # 因为输入x_t 的最后一个维度并不是config.vocab_size
            # 原因： 这里并不需要x_t 最后一个维度是config.vocab_size,  嵌入层会自动的将整数转换为嵌入表示，也就是在后面增加一个维度，为 emb_dim
            x_t = self.model.embeds(x_t)

            final_dist, h_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(x_t,
                                                h_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            target = target_batch[:, t]

            log_probs = torch.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction='none',
                                   ignore_index=self.pad_id)

            step_losses.append(step_loss)

            # final_dist：[batch_size, config.vocab_size + batch.max_art_oovs]
            # 对得到的结果在第二个维度进行采样，将采样的数量设置为1. 返回的结果是采样的位置。
            # x_t : [batch_size, 1] --> [batch_size]
            x_t = torch.multinomial(final_dist, 1).squeeze()
            
            is_oov = (x_t >= config.vocab_size).long()
            # x_t: [batch_size]
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id

        losses = torch.sum(torch.stack(step_losses, 1), 1)

        batch_avg_loss = losses / dec_lens
        mle_loss = torch.mean(batch_avg_loss)

        return mle_loss

    # 一步迭代进行的所有的步骤
    def train_one_batch(self, batch):

        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, \
            extra_zeros, context = get_enc_data(batch)

        enc_batch = self.model.embeds(enc_batch)

        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

        if self.opt.train_mle == 'yes':
            mle_loss = self.train_batch_MLE(enc_out, enc_hidden, enc_padding_mask,
                                            context, extra_zeros, enc_batch_extend_vocab, batch)
        else:
            mle_loss = get_cuda(torch.FloatTensor([0]))

        self.trainer.zero_grad()
        mle_loss.backward()
        self.trainer.step()

        return mle_loss.item()

    # 真正的train的迭代部分
    def train_iters(self):
        iter = self.setup_train()
        count = mle_total = 0

        while iter <= config.max_iterations:
            batch = self.batcher.next_batch()
            try:
                mle_loss = self.train_one_batch(batch)
            except KeyboardInterrupt:
                print("-------------Keyboard Interrupt------------")
                exit(0)

            mle_total += mle_loss
            mle_loss = 0
            count += 1
            iter += 1

            if iter % 1000 == 0:
                mle_avg = mle_total / count

                print('iter:', iter, 'mle_loss:', "%.3f" % mle_avg)

                count = mle_total = 0
                sys.stdout.flush()

            if iter % 2000 == 0:
                self.save_model(iter)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_mle', type=str, default='yes')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--new_lr', type=float, default=None)

    opt = parser.parse_args()

    print('Training mle: %s',  opt.train_mle)
    print('intra_encoder:', config.intra_encoder, 'intra_decoder', config.intra_decoder)
    sys.stdout.flush()

    train_processor = Train(opt=opt)
    train_processor.train_iters()
        










