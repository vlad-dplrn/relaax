import itertools
import logging
import numpy as np
import tensorflow as tf

from relaax.common.python.config.loaded_config import options


log = logging.getLogger(__name__)


def discounted_reward(rewards, gamma):
    # take 1D float array of rewards and compute discounted reward
    rewards = np.vstack(rewards)
    discounted_reward = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(xrange(rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_reward[t] = running_add
    # size the rewards to be unit normal
    # it helps control the gradient estimator variance
    #discounted_reward -= np.mean(discounted_reward)
    #discounted_reward /= np.std(discounted_reward) + 1e-20

    return discounted_reward


def choose_action(probabilities, exploit=False):
    if exploit:
        return np.argmax(probabilities)   # need to set greedily param
    return np.random.choice(len(probabilities), p=probabilities)


def assemble_and_show_graphs(*graphs):
    for graph in graphs:
        graph()
    log_dir = options.get("agent/log_dir", "log")
    log.info(('Writing TF summary to %s. '
              'Please use tensorboad to watch.') % log_dir)
    tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


class Utils(object):
    @staticmethod
    def map(v, mapping):

        def map_(v):
            if isinstance(v, (tuple, list)):
                return [map_(v1) for v1 in v]
            if isinstance(v, dict):
                return {k: map_(v1) for k, v1 in v.iteritems()}
            return mapping(v)

        return map_(v)

    @classmethod
    def flatten(cls, v):
        if isinstance(v, (tuple, list)):
            for vv in v:
                for vvv in cls.flatten(vv):
                    yield vvv
        elif isinstance(v, dict):
            for vv in v.itervalues():
                for vvv in cls.flatten(vv):
                    yield vvv
        else:
            yield v

    @classmethod
    def reconstruct(cls, v, pattern):
        i = iter(v)
        result = cls.map(pattern, lambda v: next(i))
        try:
            next(i)
            assert False
        except StopIteration:
            pass
        return result

    @classmethod
    def izip(cls, v1, v2):
        if isinstance(v1, (tuple, list)):
            assert isinstance(v2, (tuple, list))
            assert len(v1) == len(v2)
            for vv1, vv2 in itertools.izip(v1, v2):
                for vvv1, vvv2 in cls.izip(vv1, vv2):
                    yield vvv1, vvv2
        elif isinstance(v1, dict):
            assert isinstance(v2, dict)
            assert len(v1) == len(v2)
            for k1, vv1 in v1.iteritems():
                vv2 = v2[k1]
                for vvv1, vvv2 in cls.izip(vv1, vv2):
                    yield vvv1, vvv2
        else:
            yield v1, v2