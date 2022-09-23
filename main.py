from proposed_train import *
from tensorboardX import SummaryWriter

model_cfg = './config/proposed.cfg'
model_weight = None

save_path = './weights'

writer = SummaryWriter()
train_p(model_cfg, model_weight, '/home/kang/data/', save_path, epochs = 200, tb_writer = writer)


