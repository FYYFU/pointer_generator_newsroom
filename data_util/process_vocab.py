import glob
from data_util import config
from data_util import data
import torch

# 目前只考虑article，将title不纳入考量，
# 未来可以将title和article整合到一起。

def process_dataset(file_path):

    file_name = file_path.split('/')[-1]

    # 这个writer的主要作用将title从原始数据集中去掉，并且去掉<s> </s> 等字符。
    file_writer = open(config.vocab_path + "/" + 'summary_' + file_name, 'w')
    file_reader = open(file_path, 'r')

    for line in file_reader:
        try:
            total = line.strip().split('<sec>')
            title, summary, article = total[0], total[1], total[2]
            summary = ' '.join(data.summary2sents(summary))
            article = ' '.join(data.summary2sents(article))
        except:
            continue

        file_writer.write('<sec>'.join([summary.lower(), article.lower()]) + '\n')

    file_reader.close()
    file_writer.close()

def get_vocab():
    file_list = glob.glob(config.vocab_path + 'summary_*')
    vocab = {}

    for file_path in file_list:

        file_reader = open(file_path, 'r')

        for line in file_reader:
            try:
                total = line.strip().split('<sec>')
                summary, article = total[0], total[1]
                summary_words = summary.split(' ')
                article_words = article.split(' ')
            except:
                continue

            tokens = summary_words + article_words
            tokens = [t.strip() for t in tokens]
            tokens = [t for t in tokens if t != '']

            for wd in tokens:
                try:
                    vocab[wd] += 1
                except:
                    vocab[wd] = 1

    vocab_list = [[wd, vocab[wd]] for wd in vocab]
    # 最后面的[::-1]的作用是将list颠倒过来，因为sort默认为升序排列。所以要转化为降序排列。
    vocab_list = sorted(vocab_list, key=lambda k: k[1])[::-1]

    return vocab_list

# if __name__ == '__main__':

    # file_list = glob.glob(config.vocab_path + '*.txt')
    #
    # for file_path in file_list:
    #     process_dataset(file_path)
    #vocab_List = get_vocab()

    # my = torch.load('/home/yuf/pointer_generator/data_vocab/process_vocab.vocab')
    #
    # final_vocab_path = config.vocab_path + "/final_vocab.vocab"
    #
    # final_vocab_writer = open(final_vocab_path, 'w')
    #
    # for index, item in enumerate(my):
    #
    #     if index <= config.vocab_size:
    #         final_vocab_writer.write(item[0] + ' ' + str(item[1]) + '\n')
    #     else:
    #         break
    #
    # print("finish process vocab. vocab_size:", config.vocab_size)

    # final_vocab_path = config.vocab_path
    # final_small_vocab_path = "/home/yuf/pointer_generator/data_vocab/small_vocab.vocab"
    #
    # reader = open(final_vocab_path, 'r')
    # writer = open(final_small_vocab_path, 'w')
    #
    # for index, line in enumerate(reader.readlines()):
    #
    #     if index < config.vocab_size:
    #         writer.write(line.strip() + '\n')
    #     else:
    #         break
    #
    # reader.close()
    # writer.close()














