# # 导入包和版本查询
#
# import torch
# import torch.nn as nn
#
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.get_device_name(0))
#
# # 可复现性，在同一个设备上应该保证可复现性。
# # 在程序开始的时候，固定torch的随机种子，同时把numpy的随机种子也固定。
#
# import numpy as np
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
#
# # 张量的基本信息
# tensor = torch.randn(3, 4, 5)
# print(tensor.type())
# print(tensor.size())
# print(tensor.dim())
#
# # 命令张量。在pytorch1.3之后可以对不同维度进行命名操作
# NCHW = ['N', 'C', 'H', 'W']
# images = torch.randn(32, 3, 56, 56, names=NCHW)
# images.sum('C')
# images.select('C', index=0)
#
# tensor = torch.randn(3, 4, 1, 2, names=('C', 'N', 'H', 'W'))
# print(tensor.size())
# # 使用align_to 方法对维度进行排序。
# tensor = tensor.align_to('N', 'C', 'H', 'W')
# print(tensor.size())
#
# #torch.set_default_dtype(torch.FloatTensor)
#
# tensor = tensor.cuda()
# tensor = tensor.cpu()
# tensor = tensor.float()
# tensor = tensor.long()
#
# # torch.Tensor 和 np.ndarray之间的转换。
# # 除了charTensor之外，其他所有的CPU上的张量都支持转换为numpy格式然后再转化回来，
#
# ndarray = tensor.cpu().numpy()
# tensor = torch.from_numpy(ndarray).float()
# tensor = torch.from_numpy(ndarray.copy()).float()
#
# # 从只包含一个元素的张量中获得这个元素的值，可以直接使用item()方法。
#
# value = torch.rand(1).item()
#
# tensor = torch.rand(2, 3, 4)
# shape = (6, 4)
# tensor = torch.reshape(tensor, shape)
#
# reshap = torch.randperm(tensor.size(0))
# print(tensor)
# tensor = tensor[reshap]
# print(tensor)

import torch
import numpy





