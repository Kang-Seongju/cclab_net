from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
from PIL import Image
from PIL import ImageFile
from utils.utils import read_class
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import json

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)

def ms_coco(anno_path, sizes):

    size = float(sizes)

    classes = {}
    dic = {}
    img_bbox = {}
    img_path = []
    with open(anno_path, 'r') as cf:
        json_data = json.load(cf)

    image = json_data['images']
    for imgs in image:
        dic[imgs['id']]= [imgs['file_name'], imgs['height'], imgs['width']]

        img_bbox[imgs['file_name']] = []
        img_path.append(imgs['file_name'])

    nc = len(json_data['categories'])
    for i in range(0, nc, 1):
        classes[json_data['categories'][i]['id']] = json_data['categories'][i]['name']
        
    anno = json_data['annotations']
    for an in anno:
        cls_id = an['category_id']
        img_name = dic[an['image_id']][0]

        height = int(dic[an['image_id']][1])
        width = int(dic[an['image_id']][2])

        x1 = int(an['bbox'][0])
        y1 = int(an['bbox'][1])
        w = int(an['bbox'][2])
        h = int(an['bbox'][3])

        wp = float(width) / size / size
        hp = float(height) / size / size

        cx = x1 + w / 2
        cy = y1 + h / 2

        img_bbox[img_name].append([cls_id, cx * wp, cy * hp, w * wp, h * hp])

    '''
    classes = [id, category]
    img_bbox = dictionary [id_string] ->list [cls, cx, cy, w, h] with normalization as 0~1
    img_path = file name list
    '''
    return classes, img_bbox, img_path

class ListDataset(Dataset):
    def __init__(self, anno_path, file_path, transform, img_size, multiscale, cls):
        super(ListDataset, self).__init__()

        self.anno_path = anno_path
        self.file_path = file_path
        self.img_size = img_size
        self.cls = read_class(cls)

        self.img_files = []
        self.bbox_list = []

        # anno_file_path, image_root_path /, class list
        cls_dic, img_bbox, img_path = ms_coco(anno_path, img_size)

        for img in img_path:
            self.img_files.append(file_path + img)
            boxs = []
            for box in img_bbox[img]:
                custom_cls_id = self.cls.index(cls_dic[box[0]])
                boxs.append([custom_cls_id, box[1], box[2], box[3], box[4]])

            self.bbox_list.append(boxs)

        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = round(self.img_size / 32 / 1.5)
        self.max_size = round(self.img_size / 32 * 1.5)
        self.batch_count = 0
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            boxes = self.bbox_list[index]
            boxes = np.array(boxes)
            boxes = boxes.reshape(-1, 5)
        except Exception:
            print(f"Could not read label.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('error' + dir)
