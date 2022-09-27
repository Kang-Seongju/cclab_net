from __future__ import division
from parser import parse_arguments
from train import *
import os
import sys

class PATH():
    def __init__(self):
        self.ROOT_DIR = '/home/cclab'
        self.COCO_DIR = os.path.join(self.ROOT_DIR, 'coco')
        self.VAL_DIR = os.path.join(self.COCO_DIR,'images','val2017')
        self.model_save_path = os.path.join(self.ROOT_DIR,'weights')

if __name__ == '__main__':
    args = parse_arguments()
    
    path = PATH()
    print(path.ROOT_DIR)
    train_model(args, path)