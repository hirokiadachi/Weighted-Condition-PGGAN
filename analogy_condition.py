#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import argparse
import os
from PIL import Image
import chainer
from chainer import serializers
from chainer import Variable
from chainer import cuda
#import dataset
import network
import utils
    
def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--gen', type=str, default=None)
    parser.add_argument('--depth', '-d', type=int, default=0)
    parser.add_argument('--out', '-o', type=str, default='img/')
    parser.add_argument('--num', '-n', type=int, default=10)
    args = parser.parse_args()

    male = [0, 0, 0, 1, 0]

    gen = network.Generator(depth=args.depth)
    print('loading generator model from ' + args.gen)
    serializers.load_npz(args.gen, gen)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        gen.to_gpu()

    xp = gen.xp
        
    #xp.random.seed(seed=44)
    z = gen.z(1)
    #z2 = gen.z(1)
    attr = xp.asarray([male], dtype=xp.float32)
    #atte = Variable(attr)
    #z = Variable(z)

    #attr_num = [4, 3, 2, 1, 0]
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    f = 0
    for i in range(5):
        for j in range(args.num):
            p = j/(args.num-1)
            if i != 3:
                attr[0][i] = p
            else:
                attr[0][i] = 1 - p 
            #print(attr)
            #c = xp.reshape(c, (1, c.shape[0]))
            x = gen(z, attr, alpha=1.0)
            x = chainer.cuda.to_cpu(x.data)

            img = x[0].copy()
            Attr_dir = '/Attr_{}'.format(i)
            if not os.path.exists(args.out+Attr_dir):
                os.makedirs(args.out+Attr_dir)
            filename = os.path.join(args.out+Attr_dir, 'gen_{}.png'.format(j))
            utils.save_image(img, filename)

if __name__ == '__main__':
    generate()
