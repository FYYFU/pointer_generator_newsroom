from get_max_enc_dec_step import get_len

train_data_path = '/home/yuf/pointer_generator/data/9news.com.au.txt'
valid_data_path = '/home/yuf/pointer_generator/data/abcnews.go.com.txt'
test_data_path = '/home/yuf/pointer_generator/data/abcnews.go.com.txt'

save_model_path = '/home/yuf/pointer_generator/saved_model/'
vocab_path = '/home/yuf/pointer_generator/data_vocab/final_vocab.vocab'

hidden_dim = 512
emb_dim = 256
# 训练时候的batch 设置为8
# 测试时候的batch 设置为128
batch_size = 64

max_enc_steps,max_dec_steps = get_len(data_path=train_data_path)

# max_enc_steps = 30
# max_dec_steps = 30

beam_size = 4
min_dec_steps = 3
vocab_size = 100000

lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 500000

intra_encoder = True
intra_decoder = True