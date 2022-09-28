from __future__ import division
from parser import parse_arguments
from train import *
import os
import sys


class PATH():
    def __init__(self, dataset):
        self.ROOT_DIR = '/home/kang'
        self.dataset = dataset
        self.CUR_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.DIR = os.path.join(self.ROOT_DIR, dataset)
        self.TRAIN_IMG_DIR = os.path.join(self.DIR, 'images', 'train')
        self.VAL_IMG_DIR = os.path.join(self.DIR,'images','val')
        self.TRAIN_LAB_DIR = os.path.join(self.DIR, 'labels', 'train')
        self.VAL_LAB_DIR = os.path.join(self.DIR, 'labels', 'val')
        self.NAMES_DIR = os.path.join(self.DIR, dataset+'.names')
        self.model_save_path = os.path.join(self.CUR_DIR, dataset, 'weights')
        

if __name__ == '__main__':
    args = parse_arguments()
    path = PATH('visdrone')
    train_model(args, path)