from model.Encoder import Encoder
from model.Decoder import Decoder

import torch.nn as nn
from data_util import init_wt, config
from train_utils import get_cuda


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embeds = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt.init_wt_normal(self.embeds.weight)

        self.encoder = get_cuda(self.encoder)
        self.decoder = get_cuda(self.decoder)
        self.embeds = get_cuda(self.embeds)


# if __name__ == '__main__':
#
#     my_model = Model()
#     my_model_paramters = my_model.parameters()
#
#     print(my_model_paramters)
#     my_model_paramters_group = list(my_model_paramters)
#     print(my_model_paramters_group)