from data_util import data
import glob

import matplotlib.pyplot as plt
import matplotlib

"""
根据传入的dict 的key 对dict进行升序排列。
dict的key和value都是int，分别代表了article的长度，和这个长度的article在整个文件中的个数。
"""

def get_data_from_dict(dict):

    my_keys = []
    my_values = []
    for i in dict.keys():
        my_keys.append(i)

    my_keys.sort()

    for i in my_keys:
        my_values.append(dict[i])

    return my_keys, my_values

"""
用来得到绘制百分比曲线图的数据, 得到百分比图的纵坐标。
"""
def process_to_get_curve(values):

    percent = []
    total = 0.
    for i in values:
        total += i

    now = 0
    for i in values:
        now += i
        percentage = now * 1. / total
        percent.append(percentage)

    return percent

"""
下面这两个方法是用来获得一个给定得数据集文档中article 和 summary 
"""

def get_len(data_path, percentage=0.9):

    _, _, article_keys, \
    article_values, summary_keys, summary_values = get_max_step_and_lens(data_path)

    article_percent = process_to_get_curve(values=article_values)
    summary_percent = process_to_get_curve(values=summary_values)

    enc_max_len = get_percent_step(article_keys=article_keys, percent=article_percent, percentage=percentage)
    dec_max_len = get_percent_step(article_keys=summary_keys, percent=summary_percent, percentage=percentage)

    return enc_max_len, dec_max_len

def get_percent_step(article_keys, percent, percentage=0.9):

    for index, item in enumerate(percent):
        if item < percentage:
            continue
        else:
            percentage_step = article_keys[index]
            return percentage_step

def get_max_step_and_lens(file):

    enc_max_len = 0
    dec_max_len = 0

    article_lens_dict = {}
    summary_lens_dict = {}

    with open(file, 'r') as f:
        for line in f.readlines():
            total = line.strip().split('<sec>')
            title, summary, article = total[0], total[1], total[2]
            summary_words = ' '.join(data.summary2sents(summary))
            article_words = ' '.join(data.summary2sents(article))

            summary_len = len(summary_words.split())
            article_len = len(article_words.split())

            if article_len not in article_lens_dict.keys():
                article_lens_dict.update({article_len: 1})
            else:
                article_lens_dict[article_len] += 1

            if summary_len not in summary_lens_dict.keys():
                summary_lens_dict.update({summary_len: 1})
            else:
                summary_lens_dict[summary_len] += 1

            if summary_len > dec_max_len:
                dec_max_len = summary_len
            if article_len > enc_max_len:
                enc_max_len = article_len

    article_keys, article_values = get_data_from_dict(article_lens_dict)
    summary_keys, summary_values = get_data_from_dict(summary_lens_dict)

    return enc_max_len, dec_max_len, article_keys, article_values, summary_keys, summary_values

"""
绘制曲线图，并且标出给定的地点。
x: 所有数据的横坐标组成的列表。
y: 所有数据的纵坐标组成的列表。
percentage: 想要在图上标示出来的百分比点.[0, 1]
data_path: 对应的文件的路径，用来解析出来domain名称，从而在保存图像的时候应用。
"""
def plot_percent_curve(x, y, percentage, data_path):

    fig = plt.figure()

    point_x, point_y = 0., 0.

    for index, item in enumerate(y):
        if item < percentage:
            continue
        else:
            point_x = x[index]
            point_y = item
            break

    plt.plot(x, y)
    plt.plot([point_x,point_x], [0 ,point_y],'k--',linewidth = 1.)
    plt.plot([0, point_x], [point_y, point_y], 'k--', linewidth=1.)

    file_name = data_path.split('/')[-1]

    plt.title(file_name + " percentage curve")

    plt.ylim((0, 1))
    plt.xlim(0)
    plt.xlabel('article len')
    plt.ylabel('percent')

    plt.annotate("(%s, %s)" %(point_x, point_y), xy=(point_x, point_y), xytext=(-20, 10), textcoords='offset points')

    plt.savefig('/home/yuf/pointer_generator/data_distribution_picture/' + file_name + '_curve.png')
    plt.show()

"""
输出柱状图
"""
def plot_number_bar(x, y, data_path):

    plt.bar(x, y)
    file_name = data_path.split('/')[-1]

    plt.title(file_name + ' bar')
    plt.xlabel('article len')
    plt.ylabel('article number')
    plt.xlim(0)
    plt.ylim(0)

    plt.savefig('/home/yuf/pointer_generator/data_distribution_picture/' + file_name + '_bar.png')
    plt.show()


if __name__ == '__main__':

    file_list = glob.glob('/home/yuf/pointer_generator/data/*')

    # data_path = file_list[0]
    #
    # enc_max_len, dec_max_len, article_keys, \
    # article_values, summary_keys, summary_values = get_max_step_and_lens(data_path)
    #
    # article_percent = process_to_get_curve(article_values)
    #
    # plot_number_bar(article_keys, article_values, data_path=data_path)
    # plot_percent_curve(article_keys, article_percent, percentage=0.9, data_path=data_path)

    # for data_path in file_list:
    #
    #     enc_max_len, dec_max_len, article_keys, \
    #     article_values, summary_keys, summary_values = get_max_step_and_lens(data_path)
    #
    #     article_percent = process_to_get_curve(article_values)
    #
    #     plot_number_bar(article_keys, article_values, data_path=data_path)
    #     plot_percent_curve(article_keys, article_percent, percentage=0.9, data_path=data_path)
    #     print('finished ' + data_path.split('/')[-1])
    #
    # print('all finished!')

    # enc_max_len, dec_max_len = get_len('/home/yuf/pointer_generator/data/9news.com.au.txt')
    #
    # print(enc_max_len, dec_max_len)






