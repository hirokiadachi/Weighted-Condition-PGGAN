#!/usr/bin/env python3
import os
import glob
import shutil
import utils
import numpy as np
from tqdm import tqdm

from chainer.datasets import tuple_dataset
from PIL import Image



def random_box(size, d):
    l = np.random.randint(0, d+1)
    t = np.random.randint(0, d+1)
    r = size[0] - (d-l)
    b = size[1] - (d-t)
    return (l, t, r, b)

def get_example(item, depth, i, dir_path='resized_images/'):
    imgData = []
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


    for index, path in tqdm(enumerate(item)):
        img = Image.open(path)
        img = img.crop(random_box(img.size, 16))
        
        height = 2**(2+depth)
        width  = 3*(2**depth)
        img = img.resize((width, height))
        img = np.array(img, dtype=np.float32)/256
        if len(img.shape) == 2:
            img = np.broadcast_to(img, (3, img.shape[0], img.shape[1]))
        else:
            img = np.transpose(img, (2, 0, 1))
        save_name = dir_path+'{:0>6}.jpg'.format(index+1)
        utils.save_image(img, save_name)
        #imgData.append(img)

    return imgData

def labelandID(dir_name, file_name, choice, depth):
    print('==================================================')
    print('start to join label and pixel of image....')


    with open(file_name, 'r') as f:
        lines = f.readlines()
   
    Attr = []
    n_labels = 2
    lines = lines[0:choice]
    for line in tqdm(lines):
        line = line.split()
        attr = np.asarray(line, dtype=np.int32)
        #attr = np.eye(n_labels)[line]
        Attr.append(attr)

    new_attr = np.asarray(Attr, dtype=np.float32)
    attr_num = [20, 9, 15, 24, 31] ## Male, Blond_Hair, Eyeglasses, No_Beard, Smiling
    for num in attr_num:
        attribute = new_attr[:, num]
        attribute = attribute.T
        try:
            labels = np.vstack((labels, attribute))
        except:
            labels = attribute
    labels = labels.T
    
    print('Label shape: {}\nMake Attribute Label finished.'.format(labels.shape))

    item = glob.glob(dir_name + '/*')
    item = sorted(item)
    item = item[0:choice]

    imgData = get_example(item, depth, i=None)
    print('Make image finished.') 
    print('==================================================')

    resize_image_path = os.listdir('resized_images')
    resize_image_path = sorted(resize_image_path)
    train = [(im_path, label) for im_path , label in zip(resize_image_path, labels)]
    
    return train

if __name__ == '__main__':
    file_name = './text_file/out_attr.txt'
    dir_name  = './CelebA_zoom'
    choice    = 100000
    batchsize = 40
    n_hidden  = 100
    train = labelandID(dir_name, file_name, choice, batchsize, n_hidden)
