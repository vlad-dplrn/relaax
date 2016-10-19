import numpy as np
import random

from accum_trainer import AccumTrainer
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from params import *
from keras.optimizers import RMSprop


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index, session,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        if USE_LSTM:
            self.local_network = GameACLSTMNetwork(ACTION_SIZE, session, thread_index, device)
        else:
            self.local_network = GameACFFNetwork(ACTION_SIZE, session, device)

        self.local_network.prepare_loss(ENTROPY_BETA)
        '''
        optimizer = RMSprop(lr=initial_learning_rate,   # should be init before pass
                            rho=RMSP_ALPHA,             # 0.9 --> 0.99
                            epsilon=RMSP_EPSILON,       # 1e-8 --> 0.1
                            # decay=0.0,                # old keras ver doesn't support this
                            clipnorm=GRAD_NORM_CLIP)    # 40.0
        self.local_network.net.compile(loss=loss, optimizer=optimizer)'''

        # TODO: don't need accum trainer anymore with batch
        self.trainer = AccumTrainer(device)
        self.trainer.prepare_minimize(self.local_network.total_loss,
                                      self.local_network.get_vars())

        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.trainer.get_accum_grad_list())

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GameState(113 * thread_index)

        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step)\
                        / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        values = []
        sum = 0.0
        for rate in pi_values:
            sum = sum + rate
            value = sum
            values.append(value)

        r = random.random() * sum
        for i in range(len(values)):
            if values[i] >= r:
                return i
        # fail safe
        return len(values) - 1

    def process(self, sess, global_t):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        # reset accumulated gradients
        sess.run(self.reset_gradients)
        # self.local_network.net.optimizer.weights = []

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        if USE_LSTM:
            start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            action = self.choose_action(pi_)

            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print("pi=", pi_)
                print(" V=", value_)

            # process game
            self.game_state.process(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                print("score=", self.episode_reward)

                self.episode_reward = 0
                self.game_state.reset()
                if USE_LSTM:
                    self.local_network.reset_state()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)

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
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if USE_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            sess.run(self.accum_gradients,
                     feed_dict={
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R,
                         self.local_network.initial_lstm_state: start_lstm_state,
                         self.local_network.step_size: [len(batch_a)]})
        else:
            sess.run(self.accum_gradients,
                     feed_dict={
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R})

        cur_learning_rate = self._anneal_learning_rate(global_t)

        sess.run(self.apply_gradients,
                 feed_dict={self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (self.local_t % 100) == 0:
            print("TIMESTEP", self.local_t)

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
