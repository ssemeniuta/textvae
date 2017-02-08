import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


class SGD(object):

    def __init__(self, lr):
        self.lr = theano.shared(np.cast[np.float32](lr), name="lr")
        self.vars = [self.lr]

    def __call__(self, params, grads):
        updates = OrderedDict()
        for param, grad in zip(params, grads):
            updates[param] = param - self.lr * grad
        return updates


class Momentum(object):

    def __init__(self, lr, mom=0.9):
        self.sgd = SGD(lr)
        self.mom = mom
        self.lr = self.sgd.lr
        self.vars = self.sgd.vars

    def __call__(self, params, grads):
        updates = self.sgd(params, grads)

        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable,
                                     name=param.name + "_v")
            self.vars.append(velocity)
            x = self.mom * velocity + updates[param]  # x == mom * v_t-1 + p_t-1 - lr * grad
            updates[velocity] = x - param  # v_t == mom * v_t-1 - lr * grad
            updates[param] = x  # p_t == p_t-1 + v_t

        return updates


class RMSProp(object):

    def __init__(self, lr=1.0, rho=0.9, epsilon=1e-6):
        self.lr = theano.shared(np.cast[np.float32](lr), name="lr")
        self.rho = rho
        self.eps = epsilon
        self.grad_l2 = 0
        self.scaled_grad_l2 = 0
        self.vars = [self.lr]

    def __call__(self, params, grads):
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable, name=param.name+"_rms")
            accu_new = self.rho * accu + (1 - self.rho) * grad ** 2
            updates[accu] = accu_new
            updates[param] = param - self.lr * grad / T.sqrt(accu_new + self.eps)
            self.grad_l2 += T.sum(grad ** 2)
            self.scaled_grad_l2 += T.sum((grad / T.sqrt(accu_new + self.eps)) ** 2)
            self.vars.append(accu)

        self.grad_l2 = T.sqrt(self.grad_l2)
        self.scaled_grad_l2 = T.sqrt(self.scaled_grad_l2)
        return updates

    def additional_info(self):
        return [self.grad_l2, self.scaled_grad_l2]


class AdaDelta(object):

    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6):
        self.lr = theano.shared(np.cast[np.float32](lr))
        self.rho = rho
        self.eps = epsilon

    def __call__(self, params, grads):
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)

            # update accu (as in rmsprop)
            accu_new = self.rho * accu + (1 - self.rho) * grad ** 2
            updates[accu] = accu_new

            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + self.eps) /
                      T.sqrt(accu_new + self.eps))
            updates[param] = param - self.lr * update

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = self.rho * delta_accu + (1 - self.rho) * update ** 2
            updates[delta_accu] = delta_accu_new

        return updates


class Adam(object):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = theano.shared(np.cast[np.float32](learning_rate), name="lr")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.step_l2 = 0
        self.vars = [self.lr]

    def __call__(self, params, grads):
        t_prev = theano.shared(np.cast[np.float32](0.), name="t_prev")
        updates = OrderedDict()

        t = t_prev + 1
        a_t = self.lr*T.sqrt(1-self.beta2**t)/(1-self.beta1**t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable,
                                   name=param.name + "_m")
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable,
                                   name=param.name + "_v")
            self.vars.append(m_prev)
            self.vars.append(v_prev)

            m_t = self.beta1*m_prev + (1-self.beta1)*g_t
            v_t = self.beta2*v_prev + (1-self.beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + self.eps)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step
            self.step_l2 += T.sqrt(T.sum(step ** 2))

        updates[t_prev] = t
        self.vars.append(t_prev)
        return updates

    def additional_info(self):
        return [self.step_l2]
