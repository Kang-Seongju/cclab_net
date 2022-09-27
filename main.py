from __future__ import division
from parser import parse_arguments
from train import *
import os
import sys

class PATH():
    def __init__(self):
        self.ROOT_DIR = '/home/kang'
        self.COCO_DIR = os.path.join(self.ROOT_DIR, 'coco')
        self.model_save_path = os.path.join(self.ROOT_DIR,'weights')

if __name__ == '__main__':
    args = parse_arguments()
    
    path = PATH()
    print(path.ROOT_DIR)
#  def train(args, model_cfg, device, tb_writer, path, mixed_precision):
    train_model(args, args.cfg, path)