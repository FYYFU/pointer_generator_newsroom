# from rouge import Rouge
#
# hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
#
# reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
#
# rouge = Rouge()
#
# scores = rouge.get_scores(hypothesis, reference)
#
# print(scores)
#
# import torch
#
# h = torch.ones([2, 3])
# c = torch.zeros([2, 3])
#
# print(h * c)
#
# import glob
#
# filelist = glob.glob('/home/yuf/Text-Summarizer-Pytorch-master/data/unfinished/*')
# print(filelist)

import matplotlib.pyplot as plt
import re
import logging as log
from prettytable import PrettyTable

log.basicConfig(level=log.INFO)


def plot_trained_model_result():
    data_path = '/home/yuf/pointer_generator/saved_model/result.txt'
    with open(data_path, 'r') as f:
        x_axis = []
        y_axis = []
        for index, line in enumerate(f.readlines()):
            result = line.strip().split(" ")
            if index >= 2:
                x_axis.append(float(result[1]))
                y_axis.append(float(result[3]))

    point_y = 10.
    point_x = 0
    for index, value in enumerate(y_axis):
        if value < point_y:
            point_y = value
            point_x = x_axis[index]

    plt.plot(x_axis, y_axis)
    plt.plot([point_x, point_x], [0, point_y], 'k--')
    plt.plot([0, point_x], [point_y, point_y], 'k--')
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("iter")
    plt.ylabel('loss')
    plt.savefig('./result.png')
    plt.show()

    log.info('trained model result: {:g}, {:g}' .format(point_x, point_y))


def plot_domain_test_result(data_path):
    domain_socre = {}
    with open(data_path, 'r') as f:

        for index, line in enumerate(f.readlines()):
            if index < 3:
                 continue
            else:
                if index % 2 != 0:
                    domain = re.split("[/.]", line.strip())[5:-1]
                    domain = " ".join(domain)
                    domain_socre[domain] = []
                else:
                    domain_socre[domain].append(re.split("[:,]", line.strip())[3])
                    domain_socre[domain].append(re.split("[:,]", line.strip())[10])
                    domain_socre[domain].append(re.split("[:,]", line.strip())[17])
    return domain_socre

def plot_result_table(domain_scores):

    table = PrettyTable()

    table.add_column('domain', ['rouge-1', 'rouge-2', 'rouge-L'])
    for key in domain_scores.keys():
        if key != '9news com au':
            continue
        else:
            table.add_column(key, domain_scores[key])

    for key in domain_scores.keys():
        if key != '9news com au':
            table.add_column(key, [float(domain_scores[key][i]) - float(domain_scores['9news com au'][i]) for i in
                                   range(len(domain_scores[key]))])
    return table

def main():
    eval_result_path = '/home/yuf/pointer_generator/saved_model/0140000tar_eval.txt'

    domain_scores = plot_domain_test_result(eval_result_path)
    table = plot_result_table(domain_scores)
    print(table)

if __name__ == '__main__':
    main()
