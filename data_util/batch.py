from . import example
from . import data, config
import numpy as np

class Batch(object):

    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        self.init_encoder_sequence(example_list)
        self.init_decoder_sequence(example_list)
        self.store_original_strings(example_list)


    def init_encoder_sequence(self, example_list):

        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        # 1 代表存在，0 代表填充。
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len

            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        self.art_oovs = [ex.article_oovs for ex in example_list]
        self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]


    def init_decoder_sequence(self, example_list):
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len

    # 存储这个原始得summary 和 article的主要目的是在 eval 的时候
    # 用来计算生成的summary 和 原始的 summary之间的ROUGE 分数。
    def store_original_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_summarys = [ex.original_summary for ex in example_list]
        self.original_article_sents = [ex.original_article_sents for ex in example_list]

