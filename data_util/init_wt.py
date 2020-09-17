from data_util import config

# 后面可以尝试一下使用正交初始化。
def init_lstm_wt(lstm):
    for name, _ in lstm.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
        elif 'bias' in name:
            # 将遗忘门的偏差设置为1
            bias = getattr(lstm, name)
            n = bias.size(0)
            start , end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

