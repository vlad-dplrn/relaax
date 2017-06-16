from __future__ import absolute_import

from builtins import object

from relaax.server.common import session

from . import trpo_config
from . import trpo_model
from .old_trpo_gae.agent import agent


class Agent(object):
    def __init__(self, parameter_server):
        self.ps = parameter_server
        self.metrics = parameter_server.metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        model = trpo_model.AgentModel()
        self.session = session.Session(model)
        self.agent = agent.Agent(trpo_config.options.algorithm, self.ps, self.session)
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):
        assert (state is None) == terminal
        if reward is not None:
            if self.agent.reward(reward):
                return None
        if state is not None:
            return self.agent.act(state)
        self.agent.reset()
        return None

    # environment is asking to reset agent
    def reset(self):
        self.agent.reset()
        return True
