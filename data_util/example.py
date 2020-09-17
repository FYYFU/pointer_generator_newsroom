from . import data
from . import config

class Example(object):

    def __init__(self, article_sentences, summary_sentences, vocab):

        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # 处理article
        article = ' '.join(article_sentences)
        article_words = article.split(' ')
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)
        # article中每个word的id组成的list。对于oovs，使用unk-id来表示。
        self.enc_input = [vocab.word2id(w) for w in article_words]

        # 处理summary,传入的summary是sents。是data.summary2sents
        summary = ' '.join(summary_sentences)
        summary_words = summary.split(' ')
        sum_ids = [vocab.word2id(w) for w in summary_words]

        self.dec_input, _ = self.get_dec_inp_targ_seqs(sum_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # 针对pointer-generator model。能够从原文中复制单词，从而扩充了vocab。
        self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
        sum_ids_extend_vocab = data.summary2ids(summary_words, vocab=vocab, article_oovs=self.article_oovs)
        _, self.target = self.get_dec_inp_targ_seqs(sum_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

        self.original_article = article
        self.original_summary = summary
        self.original_article_sents = article_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        input = [start_id] + sequence[:]
        target = sequence[:]

        if len(input) > max_len:
            input = input[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)

        assert len(input) == len(target)
        return input, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)




        