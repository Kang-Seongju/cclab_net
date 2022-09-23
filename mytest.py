import torch.nn as nn
from torch.autograd import Variable
import torch
from proposed import * 
from proposed_model import *
from models import *
from torchsummary import summary as summary_

x = torch.rand([8, 3,416,416]).to('cuda')
model = CCLAB('./config/proposed.cfg',(416,416), 'default').to('cuda')

summary_(model, (3, 416,416), batch_size = 8)