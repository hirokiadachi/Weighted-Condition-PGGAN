import numpy as np
import chainer
import chainer.functions as F
from chainer import reporter
from chainer import Variable

class WganGpUpdater(chainer.training.StandardUpdater):
    def __init__(self, alpha, delta, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.lam = 10
        self.epsilon_drift = 0.001
        self.alpha = alpha
        self.delta = delta
        super(WganGpUpdater, self).__init__(**kwargs)

    def update_core(self):
        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        xp = self.gen.xp

        # update discriminator
        batch = self.get_iterator('main').next()
        x = [batch[i][0] for i in range(len(batch))]
        c = [batch[i][1] for i in range(len(batch))]
        x = xp.array(x) / 256.
        c = xp.array(c, xp.float32)
        m = len(x)

        z = self.gen.z(m)
        x_tilde = self.gen(z, c, self.alpha).data
        
        epsilon = xp.random.rand(m, 1, 1, 1).astype('f')
        x_hat = Variable(epsilon * x + (1 - epsilon) * x_tilde)

        dis_x = self.dis(x, c, self.alpha)

        # recognission attribute
        male, hair, eye, beard, smiling = self.dis(x, None, self.alpha)
        c1 = (c[:, 0]).astype(xp.int32)
        c2 = (c[:, 1]).astype(xp.int32)
        c3 = (c[:, 2]).astype(xp.int32)
        c4 = (c[:, 3]).astype(xp.int32)
        c5 = (c[:, 4]).astype(xp.int32)
        male_loss  = F.softmax_cross_entropy(male, c1)       
        hair_loss   = F.softmax_cross_entropy(hair, c2)       
        eye_loss = F.softmax_cross_entropy(eye, c3)       
        beard_loss = F.softmax_cross_entropy(beard, c4)       
        smiling_loss = F.softmax_cross_entropy(smiling, c5)  
        recog_loss = male_loss+hair_loss+eye_loss+beard_loss+smiling_loss     
 
        loss_d = self.dis(x_tilde, c, self.alpha) - dis_x

        g_d, = chainer.grad([self.dis(x_hat, c, self.alpha)], [x_hat], enable_double_backprop=True)
        g_d_norm = F.sqrt(F.batch_l2_norm_squared(g_d) + 1e-6)
        g_d_norm_delta = g_d_norm - 1
        loss_l = self.lam * g_d_norm_delta * g_d_norm_delta
        
        loss_dr = self.epsilon_drift * dis_x * dis_x

        dis_loss = F.mean(loss_d + loss_l + loss_dr) + recog_loss

        self.dis.cleargrads()
        dis_loss.backward()
        opt_d.update()
        
        # update generator
        z = self.gen.z(m)
        x = self.gen(z, c, self.alpha)
        gen_loss = F.average(-self.dis(x, c, self.alpha)) + recog_loss

        self.gen.cleargrads()
        gen_loss.backward()
        opt_g.update()

        reporter.report({'loss_d': F.mean(loss_d), 'loss_l': F.mean(loss_l), 'loss_dr': F.mean(loss_dr), 'dis_loss': dis_loss, 'gen_loss': gen_loss, 'alpha': self.alpha})

        self.alpha = self.alpha + self.delta
