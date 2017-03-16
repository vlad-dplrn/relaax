from __future__ import print_function

import numpy as np
import random
import tensorflow as tf
import time

import relaax.algorithm_base.agent_base
import relaax.common.metrics
import relaax.common.protocol.socket_protocol

from . import network


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server

        kernel = "/cpu:0"
        if config.use_GPU:
            kernel = "/gpu:0"

        with tf.device(kernel):
            self._local_network = network.make(config)

        self.global_t = 0           # counter for global steps between all agents
        self.local_t = 0            # steps count for current agent's thread
        self.updates_t = 0          # counter for global updates
        self.episode_reward = 0     # score accumulator for current game

        self.states = []            # auxiliary states accumulator through episode_len = 0..5
        self.actions = []           # auxiliary actions accumulator through episode_len = 0..5
        self.rewards = []           # auxiliary rewards accumulator through episode_len = 0..5
        self.values = []            # auxiliary values accumulator through episode_len = 0..5

        self.episode_t = 0          # episode counter through episode_len = 0..5
        self._terminal_end = False   # auxiliary parameter to compute R in update_global and frameQueue
        self.start_lstm_state = None

        self.obsQueue = None        # observation accumulator for state = history_len * consecutive frames
        self.accum_latency = 0

        self._session = tf.Session()

        self._session.run(tf.variables_initializer(tf.global_variables()))

    def act(self, state):
        start = time.time()
        self._update_state(state)

        if self.episode_t == self._config.episode_len:
            self._update_global()

            if self._terminal_end:
                self._terminal_end = False

            self.episode_t = 0

        if self.episode_t == 0:
            # copy weights from shared to local
            self._local_network.assign_values(self._session, self._parameter_server.get_values())

            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []

            if self._config.use_LSTM:
                self.start_lstm_state = self._local_network.lstm_state_out

        pi_, value_ = self._local_network.run_policy_and_value(self._session, self.obsQueue)
        action = self._choose_action(pi_)

        self.states.append(self.obsQueue)
        self.actions.append(action)
        self.values.append(value_)

        self.accum_latency += time.time() - start

        if (self.local_t % 100) == 0:
            print("pi=", pi_)
            print(" V=", value_)
            self.metrics().scalar('server latency', self.accum_latency / 100)
            self.accum_latency = 0

        return action

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None
        self._terminal_end = True

        print("score=", self.episode_reward)
        score = self.episode_reward

        self.metrics().scalar('episode reward', self.episode_reward)
        self.episode_reward = 0

        if self._config.use_LSTM:
            self._local_network.reset_state()

        self.episode_t = self._config.episode_len
        return score

    def metrics(self):
        return self._parameter_server.metrics()

    def _reward(self, reward):
        self.episode_reward += reward

        # clip reward
        self.rewards.append(np.clip(reward, -1, 1))

        self.local_t += 1
        self.episode_t += 1
        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    @staticmethod
    def _choose_action(pi_values):
        values = []
        total = 0.0
        for rate in pi_values:
            total += rate
            value = total
            values.append(value)

        r = random.random() * total
        for i in range(len(values)):
            if values[i] >= r:
                return i
        # fail safe
        return len(values) - 1

    @staticmethod
    def _choose_act(pi_values):
        # can replace the old one similar method
        values = np.cumsum(pi_values)
        total = values[-1]
        r = np.random.rand() * total
        return np.searchsorted(values, r)

    def _update_state(self, obs):
        axis = len(obs.shape)  # extra dimension for observation
        new_obs = np.reshape(obs, obs.shape + (1,))
        if not self.terminal_end and self.local_t != 0:
            # remove oldest observation from the begining of the observation queue
            self.obsQueue = np.delete(self.obsQueue, 0, axis=axis)

            # append latest observation to the end of the observation queue
            self.obsQueue = np.append(self.obsQueue, new_obs, axis=axis)
        else:
            # copy observation several times to form initial observation queue
            self.obsQueue = np.repeat(new_obs, self._config.history_len, axis=axis)

    def _update_global(self):
        R = 0.0
        if not self.terminal_end:
            R = self._local_network.run_value(self._session, self.obsQueue)

        self.actions.reverse()
        self.states.reverse()
        self.rewards.reverse()
        self.values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accumulate gradients
        for (ai, ri, si, Vi) in zip(self.actions,
                                    self.rewards,
                                    self.states,
                                    self.values):
            R = ri + self._config.GAMMA * R
            td = R - Vi
            a = np.zeros([self._config.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if self._config.use_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            feed_dict = {
                    self._local_network.s: batch_si,
                    self._local_network.a: batch_a,
                    self._local_network.td: batch_td,
                    self._local_network.r: batch_R,
                    self._local_network.initial_lstm_state: self.start_lstm_state,
                    self._local_network.step_size: [len(batch_a)]
            }
        else:
            feed_dict = {
                    self._local_network.s: batch_si,
                    self._local_network.a: batch_a,
                    self._local_network.td: batch_td,
                    self._local_network.r: batch_R
            }

        self._parameter_server.apply_gradients(
            self._session.run(self._local_network.grads, feed_dict=feed_dict)
        )

        if (self.updates_t % 20) == 0:
            print("TIMESTEP", self.local_t)

        self.updates_t += 1
