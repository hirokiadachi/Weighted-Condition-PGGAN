import numpy as np
import chainer
import cupy
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from functions import my_sqrt, my_rsqrt

class Layer(Chain):
    def __init__(self, ch_in, ch_out, ksize1=3, ksize2=3, stride=1, pad=1, normalize=False):
        w = chainer.initializers.Normal(1.0)
        super(Layer, self).__init__(
            conv = L.Convolution2D(ch_in, ch_out, (ksize1, ksize2), stride, pad, initialW=w)
        )

        self.c = np.sqrt(2 / (ch_in * ksize1 * ksize2))
        self.eps = 1e-8

        self.normalize = normalize
        
    def __call__(self, x):
        h = x * self.c
        h = self.conv(h)
        if self.normalize:
            mean = F.mean(h*h, axis=1, keepdims=True)
            dom = my_rsqrt(mean + self.eps)
            dom = F.broadcast_to(dom, h.shape)
            h = h * dom
        return h

class Condition(Chain):
    def __init__(self, ch_in, ch_out, ksize=1):
        w = chainer.initializers.Normal(0.02)
        super(Condition, self).__init__(
            conv = L.Convolution2D(ch_in, ch_out, ksize, initialW=w),
        )

    def __call__(self, c, height, width):
        ones = cupy.ones((c.shape[0], c.shape[1], height, width), dtype=cupy.float32)
        c = cupy.reshape(c, (c.shape[0], c.shape[1], 1, 1))
        c = c*ones
        #c = c.data*ones

        h = self.conv(c)
        h = F.sigmoid(h)

        return h

class GFirstBlock(Chain):
    def __init__(self, ch_in, ch_out, concatsize):
        super(GFirstBlock, self).__init__(
            c1 = Layer(concatsize, ch_in, 4, 5, 1, 3),
            c2 = Layer(ch_in, ch_out, normalize=True),
            toRGB= Layer(ch_out, 3, ksize1=1, ksize2=1, pad=0),
            condition = Condition(5, 32),
        )

    def __call__(self, x, c, last=False):
        y0 = self.condition(c, x.shape[2], x.shape[3])
        x = F.concat((y0, x))
        l1 = F.leaky_relu(self.c1(x))
        l2 = F.leaky_relu(self.c2(l1))
        if last:
            return self.toRGB(l2)
        return l2
    
class GBlock(Chain):
    def __init__(self, ch_in, ch_out, outsize, concatsize, num):
        super(GBlock, self).__init__(
            c1 = Layer(concatsize, ch_out, normalize=True),
            c2 = Layer(ch_out, ch_out, normalize=True),
            toRGB = Layer(ch_out, 3, ksize1=1, ksize2=1, pad=0),
            condition = Condition(5, num),
        )

        self.outsize = outsize

    def __call__(self, x, c, last=False):
        l1 = F.unpooling_2d(x, 2, 2, outsize=self.outsize)
        y0 = self.condition(c, l1.shape[2], l1.shape[3])
        l1 = F.concat((y0,l1))
        l2 = F.leaky_relu(self.c1(l1))
        l3 = F.leaky_relu(self.c2(l2))
        if last:
            return self.toRGB(l3)
        return l3

class Generator(Chain):
    def __init__(self, depth):
        super(Generator, self).__init__(
            b0 = GFirstBlock(512, 512, 512+32),
            b1 = GBlock(512, 512, (8, 6),   512+32, 32),
            b2 = GBlock(512, 256, (16, 12), 512+32, 32),
            b3 = GBlock(256, 128, (32, 24), 256+32, 32),
            b4 = GBlock(128, 64, (64, 48),  128+32, 32),
            b5 = GBlock(64, 32, (128, 96),  64+32, 32),
            b6 = GBlock(32, 16, (256, 192), 32+32, 32),
        )

        self.depth = depth

    def z(self, sz):
        z = self.xp.random.randn(sz, 512, 1, 1).astype('f')
        return z

    def __call__(self, x, c, alpha=1.0):
        if self.depth > 0 and alpha < 1.0:
            h = x
            for i in range(self.depth-1):
                h = self['b%d'%i](h, c)

            h1 = self['b%d'%(self.depth-1)](h, c)
            h2 = F.unpooling_2d(h1, 2, 2, outsize=self['b%d'%self.depth].outsize)
            h3 = self['b%d'%(self.depth-1)].toRGB(h2)
            h4 = self['b%d'%self.depth](h1, c, True)
            
            h = h3 * (1 - alpha) + h4 * alpha
        else:
            h = x
            for i in range(self.depth):
                h = self['b%d'%i](h, c)

            h = self['b%d'%self.depth](h, c, True)

        return h
    
class DBlock(Chain):
    def __init__(self, ch_in, ch_out):
        super(DBlock, self).__init__(
            fromRGB = Layer(3, ch_in, ksize1=1, ksize2=1, pad=0),
            c1 = Layer(ch_in, ch_in),
            c2 = Layer(ch_in, ch_out),
        )

    def __call__(self, x, c, first=False):
        if first:
            l0 = F.leaky_relu(self.fromRGB(x))
        else:
            l0 = x
        l1 = F.leaky_relu(self.c1(l0))
        l2 = F.leaky_relu(self.c2(l1))
        l3 = F.average_pooling_2d(l2, 2, 2)
        return l3

class MinibatchStddev(Link):
    def __init__(self, ch):
        super(MinibatchStddev, self).__init__()
        
        self.eps = 1.0

    def __call__(self, x):
        mean = F.mean(x, axis=0, keepdims=True)
        dev = x - F.broadcast_to(mean, x.shape)
        devdev = dev * dev
        var = F.mean(devdev, axis=0, keepdims=True) # using variance instead of stddev
        # stddev = my_sqrt(var + self.eps)
        # stddev_mean = F.mean(stddev)
        stddev_mean = F.mean(var)
        new_channel = F.broadcast_to(stddev_mean, (x.shape[0], 1, x.shape[2], x.shape[3]))
        h = F.concat((x, new_channel), axis=1)
        return h

class DLastBlock(Chain):
    def __init__(self, ch_in, ch_out):
        super(DLastBlock, self).__init__(
            fromRGB = Layer(3, ch_in, ksize1=1, ksize2=1, pad=0),
            stddev = MinibatchStddev(ch_in+5),
            c1 = Layer(ch_in+1+5, ch_out),
            cr = Layer(ch_in, ch_out),
            c2 = Layer(ch_out, ch_out, 4, 3, 1, 0),
        )

    def __call__(self, x, c, first=False):
        if first:
            l0 = F.leaky_relu(self.fromRGB(x))
        else:
            l0 = x
        if not c is None:
            c = cupy.reshape(c, (c.shape[0], c.shape[1], 1, 1))
            ones = cupy.ones((c.shape[0], c.shape[1], l0.shape[2], l0.shape[3]), dtype=cupy.float32)
            c = ones*c
            l0 = F.concat((c,l0))
            l1 = self.stddev(l0)
            l2 = F.leaky_relu(self.c1(l1))
        else:
            l2 = F.leaky_relu(self.cr(l0))
        l3 = F.leaky_relu(self.c2(l2))
        return l3

class Discriminator(Chain):
    def __init__(self, depth):
        super(Discriminator, self).__init__(
            b1 = DBlock(16, 32),   #128
            b2 = DBlock(32, 64),   #64
            b3 = DBlock(64, 128),  #32
            b4 = DBlock(128, 256), #16
            b5 = DBlock(256, 512), #8
            b6 = DBlock(512, 512), #4
            b7 = DLastBlock(512, 512),
            l = L.Linear(512, 1),
            l1 = L.Linear(512, 512),
            #l2 = L.Linear(512, 10),
            male   = L.Linear(512, 2),
            blond  = L.Linear(512, 2),
            eye = L.Linear(512, 2),
            no_beard  = L.Linear(512, 2),
            smiling = L.Linear(512, 2),
        )

        self.depth = depth

    def __call__(self, x, c, alpha=1.0):
        if self.depth > 0 and alpha < 1:
            h1 = self['b%d'%(7-self.depth)](x, c, True)
            x2 = F.average_pooling_2d(x, 2, 2)
            h2 = F.leaky_relu(self['b%d'%(7-self.depth+1)].fromRGB(x2))
            h = h2 * (1 - alpha) + h1 * alpha
        else:
            h = self['b%d'%(7-self.depth)](x, c, True)
                
        for i in range(self.depth):
            h = self['b%d'%(7-self.depth+1+i)](h, c)

        if not c is None:
            d = self.l(h)
            d = F.flatten(d)
            return d
        else:
            d = F.relu(self.l1(h))
            #d = self.l2(d)
            male = self.male(d)
            blond = self.blond(d)
            eye = self.eye(d)
            no_beard = self.no_beard(d)
            smiling = self.smiling(d)
            return male, blond, eye, no_beard, smiling


