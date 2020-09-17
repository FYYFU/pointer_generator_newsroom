import numpy as np
import torch
from data_util import config


def get_cuda(tensor):

    if  torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

# 这个函数一共有两个目的：
# 1. 将所有的数据由原本的numpy转化为tensor
# 2. 如果使用了cuda，那么将所有的数据转化到cuda上。
def get_enc_data(batch):
    """
    返回值：
    1. enc_batch: 其中包含的输入的所有的数据。            [batch_size, max_seq_len]

    2. enc_lens：输入的batch中每个输入的长度。           [batch_size]

    3. enc_padding_mask：输入的mask情况。              [batch_size, max_seq_len]

    4. enc_batch_extend_vocab: 扩充vocab后的输入。     [batch_size, max_seq_len]

    5.extra_zeros: 保存每个article的out-of-vocabulary. [batch_size, max_art_oovs]

    6.ct_e: encoder的hidden_state.                   [batch_size, 2 * hidden_dim]
    """
    batch_size = len(batch.enc_lens)

    enc_batch = torch.from_numpy(batch.enc_batch).long()
    enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).float()

    enc_lens = batch.enc_lens

    # 后面乘以2的原因应该是这个(h, c),存储了encoder的最后的单元的隐藏状态h和记忆状态c，作为decoder的time step=0 的初始隐藏状态输入。
    ct_e = torch.zeros(batch_size, 2 * config.hidden_dim)

    enc_batch = get_cuda(enc_batch)
    enc_padding_mask = get_cuda(enc_padding_mask)
    ct_e = get_cuda(ct_e)

    # 为了实现pointer部分，需要将enc_input中的单词临时加入到vocab中。
    enc_batch_extend_vocab = None
    if batch.enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).long()
        enc_batch_extend_vocab = get_cuda(enc_batch_extend_vocab)

    # 这个部分是为了存储batch中每个article的out-of-vocabulary。
    extra_zeros = None
    if batch.max_art_oovs > 0:
        extra_zeros = torch.zeros(batch_size, batch.max_art_oovs)
        extra_zeros = get_cuda(extra_zeros)

    return enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e


# 有一点不太明白，就是这个dec_lens为什么要进行get_cuda的操作。
# enc_lens 和 enc_batch 有什么区别？
# dec_lens 和 dec_batch 有什么区别？
def get_dec_data(batch):

    dec_batch = torch.from_numpy(batch.dec_batch).long()

    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens = torch.from_numpy(batch.dec_lens).float()

    target_batch = torch.from_numpy(batch.target_batch).long()

    dec_batch = get_cuda(dec_batch)
    dec_lens = get_cuda(dec_lens)
    target_batch = get_cuda(target_batch)

    return dec_batch, max_dec_len, dec_lens, target_batch