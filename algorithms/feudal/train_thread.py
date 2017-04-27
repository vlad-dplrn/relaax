from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import gym
from scipy.misc import imresize

from networks import LocalWorkerNetwork
from config import cfg
from utils import RingBuffer


class TrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 manager_network):
        self.thread_index = thread_index
        self.manager_network = manager_network
        self.goal_buffer = RingBuffer(element_size=cfg.d,
                                      buffer_size=cfg.c)

        self.initial_learning_rate = cfg.learning_rate
        self.max_global_time_step = cfg.MAX_TIME_STEP

        self.local_network = LocalWorkerNetwork(thread_index)

        compute_gradients = tf.gradients(self.local_network.total_loss,
                                         self.local_network.weights)

        self.apply_gradients = global_network.optimizer.apply_gradients(
            zip(compute_gradients, global_network.weights)
        )

        self.sync = self.local_network.sync_from(global_network)
        self.lr = global_network.learning_rate_input

        self.env = gym.make(cfg.env_name)
        self.env.seed(113 * thread_index)
        self.state = _process_state(self.env.reset())

        self.local_t = 0
        self.episode_reward = 0

        # summary for tensorboard
        if thread_index == 0:
            self.score_input = tf.placeholder(tf.float32, [], name="episode_reward")
            tf.summary.scalar('episode_reward', self.score_input)

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate *\
                        (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    @staticmethod
    def choose_action(pi_probs):
        return np.random.choice(len(pi_probs), p=pi_probs)

    def process(self, sess, global_t, summaries, summary_writer):
        states, actions, rewards, values = [], [], [], []
        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t
        start_lstm_state = self.local_network.lstm_state_out
        manager_lstm_state = self.manager_network.lstm_state_out

        for i in range(cfg.LOCAL_T_MAX):
            z_t = sess.run(self.local_network.perception,
                         {self.local_network.s: [self.state]})
            goal, v_t = self.manager_network.run_goal_and_value(sess, z_t)

            pi_, value_ = self.local_network.run_policy_and_value(sess, self.state)
            action = self.choose_action(pi_)

            states.append(self.state)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print("TIMESTEP", self.local_t)
                print("pi=", pi_)
                print(" V=", value_)

            # act
            env_state, reward, terminal, _ = self.env.step(action)
            self.local_t += 1

            self.episode_reward += reward
            # clip reward
            rewards.append(np.clip(reward, -1, 1))

            if terminal:
                terminal_end = True
                print("Score:", self.episode_reward)

                if self.thread_index == 0:
                    summary_str = sess.run(summaries,
                                           feed_dict={self.score_input: self.episode_reward})
                    summary_writer.add_summary(summary_str, global_t)

                self.episode_reward = 0
                self.state = _process_state(self.env.reset())
                self.local_network.reset_state()    # may be move further after update
                break
            self.state = _process_state(env_state)

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.state)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accumulate gradients
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + cfg.wGAMMA * R
            td = R - Vi
            a = np.zeros([cfg.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()

        sess.run(self.apply_gradients,
                 feed_dict={
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R,
                         self.local_network.initial_lstm_state: start_lstm_state,
                         self.lr: self._anneal_learning_rate(global_t),
                         self.local_network.step_size: [len(batch_a)]})

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t


def _process_state(screen):
    resized_screen = imresize(screen, (110, 84))
    state = resized_screen[18:102, :]

    state = state.astype(np.float32)
    state *= (1.0 / 255.0)
    return state
