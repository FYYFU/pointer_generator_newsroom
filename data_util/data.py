import glob
import random
import struct
import csv

# 用来将摘要划分为句子。这两个符号不存在 vocab id。
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

# vocab file: 是一个存储所有的单词和出现在数据集中的次数的文件，
# 每一行都存储一个单词和对应的出现的次数，根据出现次数从高到低进行排序。
# Vocab类的目的就是从这个vocab_file中得到所有的word并且赋予不同的id。

class Vocab(object):

    def __init__(self, vocab_file, max_size):

        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    continue

                word = pieces[0]
                if word in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN,
                            START_DECODING, STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] '
                                    'shouldn\'t be in the vocab file, but %s is' % w)

                if word in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                self._word_to_id[word] = self._count
                self._id_to_word[self._count] = word
                self._count += 1

                # 设置词汇表的最大长度。
                if max_size != 0 and self._count >= max_size:
                    break

    def size(self):
        return self._count

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError("Id not found in vocab %d" % word_id)
        return self._id_to_word[word_id]

    # 将每个word和对应的 id存储在给定的文件中。
    def write_metadata(self, fpath):
        print('Writing word embedding metadata file to %s...' %(fpath))
        with open(fpath, 'w') as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({'word': self._id_to_word[i]})


def example_generator(data_path):

    filelist = glob.glob(data_path)
    assert filelist, ('Error: Empty filelist at %s' % data_path)

    filelist = sorted(filelist)

    for f in filelist:

        with open(f, 'r') as reader:

            for line in reader.readlines():

                total = line.split('<sec>')
                title, summary, article = total[0], total[1], total[2]

                if len(article) == 0:
                    continue
                else:
                    yield (title, summary, article)


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)

    for w in article_words:
        i = vocab.word2id(w)

        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            # This is 0 for the first article OOV, 1 for the second article OOV...
            oovs_num = oovs.index(w)
            # This is e.g. 50000 for the first article OOV, 50001 for the second...
            ids.append(vocab.size() + oovs_num)
        else:
            ids.append(i)

    return ids, oovs


def summary2ids(summary_words, vocab, article_oovs):

    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)

    for w in summary_words:
        i = vocab.word2id(w)

        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)

    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)
        except ValueError as e:
            assert article_oovs is not None, 'Error: model produced a word ID that' \
                                             'isnot in the vocab.This should not happen in baseline model'

            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:
                raise ValueError('Error: model produced word ID %i which'
                                 'corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))

        words.append(w)
    return words


def summary2sents(summary):
    cur = 0
    sents = []

    while True:
        try:
            start_p = summary.index(SENTENCE_START, cur)
            end_p = summary.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(summary[start_p+len(SENTENCE_START): end_p])
        except ValueError as e:
            return sents


def show_art_oovs(article, vocab):

    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w  for w in words]
    out_str = ' '.join(words)

    return out_str

def show_sum_oovs(summary, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = summary.split(' ')

    new_words = []

    for w in words:
        if vocab.word2id(w) == unk_token:
            if article_oovs is None:
                new_words.append('__%s__' % w)
            else:
                if w in article_oovs:
                    new_words.append('__%s__' % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:
            new_words.append(w)

    out_str = ' '.join(new_words)
    return out_str

#
# if __name__ == '__main__':
#
#     data_path = '/home/yuf/pointer_generator/data/*'
#
#     input = example_generator(data_path)
#
#     num = 0
#
#     while True:
#         try:
#             (title, summary, article) = next(input)
#         except:
#             Exception("out of data")
#             break
#
#         sentences_summary = summary2sents(summary)
#         senences_article = summary2sents(article)
#
#         print(senences_article)
#         print(sentences_summary)
#         print('')
#         num += 1
#
#         if num >= 5:
#             break








