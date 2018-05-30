# -*- coding:utf-8 -*-
"""

Generative Image with Progressive Growing GANs

"""
import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

import chainer
from network import Generator
import utils

parser = argparse.ArgumentParser(description='Image Generator')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--option', type=str, default='save')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--dirname', type=str, default='Generate_Images')
args = parser.parse_args()

attribute = [0, 1, 0, 1, 1]

gen = Generator(args.depth)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    gen.to_gpu()

chainer.serializers.load_npz('./results/gen', gen)

xp = gen.xp
#xp.random.seed(seed=1)
z = gen.z(args.num)
z = chainer.Variable(xp.asarray(z))

print('%d images Generating.....'%(args.num))

try:
    img = gen(z)
except:
    attr = [attribute for i in range(args.num)]
    attr = xp.asarray(attr, dtype=xp.float32)
    attr = attr.reshape(attr.shape[0], attr.shape[1], 1, 1)
    x = gen(z, attr, alpha=1.0)

x = chainer.cuda.to_cpu(x.data)

if not os.path.exists(args.dirname):
    os.makedirs(args.dirname)

if args.option == 'save':
    #画像を一枚ずつ保存
    for i in range(args.num):
        img = x[i].copy()
        filename = os.path.join(args.dirname, '{:0>3}.jpg'.format(i))
        utils.save_image(img, filename)

else:
    #複数の画像を並べて保存
    #parserのargs.numの値を変更した場合は下も変更する（下のコードは100枚を生成して並べて保存する）
    x = np.asarray(np.clip(x*255, 0.0, 255), dtype=np.uint8)
    _, _, H, W = x.shape
    img = x.reshape((10, 10, 3, H, W))
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape((10*H, 10*W, 3))
    x = Image.fromarray(img)
    x.save(os.path.join(args.dirname, 'facial_images.jpg'))


